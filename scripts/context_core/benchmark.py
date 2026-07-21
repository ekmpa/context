import json
import math
import os
import time
import threading
import concurrent.futures

import pandas as pd

from context_core.pipeline import Pipeline


def estimate_auto_checker_price(num_claims, checker_config_name, openai_cost=0.015):
    checker_name = checker_config_name.lower()
    if "rarr" in checker_name:
        return num_claims * (4 * openai_cost)
    if "factcheck" in checker_name:
        return num_claims * (2 * openai_cost)
    if "factool" in checker_name:
        return num_claims * (2 * (openai_cost + 0.001))
    return num_claims * (2 * openai_cost)


def sumAllObj(obj):
    return sum(obj.values())


def _env_int(name, default, minimum=1):
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return max(minimum, int(raw))
    except Exception:
        return default


def _extract_claim_counts(state):
    counts = {
        "numFalseClaims": 0,
        "numMixedClaims": 0,
        "numTrueClaims": 0,
        "numUndefinedClaims": 0,
    }

    details = state.get("detail") or state.get("log") or []
    if not isinstance(details, list):
        return counts

    for item in details:
        labels = item.get("labels")
        if isinstance(labels, list) and labels:
            normalized = []
            for label in labels:
                if isinstance(label, bool):
                    normalized.append("true" if label else "false")
                    continue
                if isinstance(label, str):
                    low = label.strip().lower()
                    if low in {"agrees", "true", "supports", "support"}:
                        normalized.append("true")
                    elif low in {"disagrees", "false", "refutes", "refute", "contradicts", "contradict"}:
                        normalized.append("false")
                    elif low in {"ambiguous", "unverifiable", "irrelevant", "unknown"}:
                        normalized.append("undefined")
                    else:
                        normalized.append("undefined")
                else:
                    normalized.append("undefined")

            uniq = set(normalized)
            if uniq == {"true"}:
                counts["numTrueClaims"] += 1
            elif uniq == {"false"}:
                counts["numFalseClaims"] += 1
            elif uniq == {"undefined"}:
                counts["numUndefinedClaims"] += 1
            else:
                counts["numMixedClaims"] += 1
            continue

        factuality = item.get("factuality")
        if isinstance(factuality, bool):
            counts["numTrueClaims" if factuality else "numFalseClaims"] += 1
        elif isinstance(factuality, (int, float)):
            counts["numTrueClaims" if factuality > 0 else "numFalseClaims"] += 1
        else:
            counts["numUndefinedClaims"] += 1

    return counts


def evaluate_free_text_with_auto_checker(
    llm_response_data, response_column_name, args, projectdir
):
    dataset_name = llm_response_data[0]["source"]
    llm_response_data = pd.DataFrame(llm_response_data)

    hot_reload = {"global_config": {}}
    if getattr(args, "openai_apikey", None):
        hot_reload["global_config"]["openai_key"] = {
            "value": args.openai_apikey,
            "env_name": "OPENAI_API_KEY",
        }
    if getattr(args, "factcheck_model", None):
        hot_reload["global_config"]["factcheck_gpt_model"] = args.factcheck_model
    if getattr(args, "rarr_model", None):
        hot_reload["global_config"]["rarr_model"] = args.rarr_model

    def _build_pipeline():
        pipeline = Pipeline(args)
        if hot_reload["global_config"]:
            pipeline.hot_reload_global_config(hot_reload)
        return pipeline

    worker_count = _env_int("RARR_SAMPLE_WORKERS", 1)
    worker_count = min(worker_count, len(llm_response_data))

    thread_local = threading.local()

    def _get_thread_pipeline():
        existing = getattr(thread_local, "pipeline", None)
        if existing is None:
            thread_local.pipeline = _build_pipeline()
        return thread_local.pipeline

    def _evaluate_index(i):
        pipeline = _get_thread_pipeline()

        prompt = llm_response_data["prompt"][i]
        response = llm_response_data["response"][i]
        sample_name = response_column_name[:-9] + f"_{dataset_name}_{i}"

        start = time.time() * 1000
        error_message = None
        state = {}
        try:
            result_label = pipeline(
                question=prompt,
                response=response,
                sample_name=sample_name,
            )
            state = getattr(pipeline, "last_state", {}) or {}
            claims = _extract_claim_counts(state)
        except Exception as exc:
            # Keep batch runs resilient: record the failure and move to next sample.
            result_label = None
            claims = {
                "numFalseClaims": 0,
                "numMixedClaims": 0,
                "numTrueClaims": 0,
                "numUndefinedClaims": 1,
            }
            error_message = str(exc)
        end = time.time() * 1000

        result = {
            "index": i,
            "start": math.floor(start),
            "end": math.floor(end),
            "llm": response_column_name,
            "dataset": llm_response_data["source"][i],
            "prompt": prompt,
            "response": response,
            "claims": claims,
            "result": result_label,
            "detail": state.get("detail", []),
        }
        if error_message is not None:
            result["error"] = error_message
        if state.get("search_failed"):
            result["discarded"] = True

        return result

    indices = list(range(len(llm_response_data["prompt"])))
    if worker_count <= 1 or len(indices) <= 1:
        entries = [_evaluate_index(i) for i in indices]
    else:
        entries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(_evaluate_index, i) for i in indices]
            for fut in concurrent.futures.as_completed(futures):
                entries.append(fut.result())
        entries.sort(key=lambda x: x.get("index", 0))

    total_entries = len(entries)
    discarded_entries = [entry for entry in entries if entry.get("discarded")]
    if discarded_entries:
        entries = [entry for entry in entries if not entry.get("discarded")]
        discarded_ratio = len(discarded_entries) / total_entries if total_entries else 0.0
        if discarded_ratio > 0.10:
            raise RuntimeError(
                f"Aborting run: {len(discarded_entries)}/{total_entries} queries failed after retries "
                f"({100.0 * discarded_ratio:.1f}%), which exceeds the 10% limit."
            )

    return entries