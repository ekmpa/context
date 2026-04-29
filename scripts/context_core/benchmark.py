import json
import math
import os
import time
from hashlib import md5

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


def _read_latest_state(truth_output_dir, sample_name):
    result_path = os.path.join(truth_output_dir, f"{sample_name}.jsonl")
    if not os.path.exists(result_path):
        return {}

    latest = None
    with open(result_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                latest = json.loads(line)

    if latest is None:
        return {}
    return latest.get("state", {})


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

    pipeline = Pipeline(args)

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
    if hot_reload["global_config"]:
        pipeline.hot_reload_global_config(hot_reload)

    for i in range(len(llm_response_data["prompt"])):
        prompt = llm_response_data["prompt"][i]
        response = llm_response_data["response"][i]
        sample_name = response_column_name[:-9] + f"_{dataset_name}_{i}"
        dirname = sample_name + "_" + md5(prompt.encode()).hexdigest()
        dirpath = os.path.join(projectdir, dirname)
        os.makedirs(dirpath, exist_ok=True)

        start = time.time() * 1000
        error_message = None
        try:
            result_label = pipeline(
                question=prompt,
                response=response,
                sample_name=sample_name,
            )
            claims = _extract_claim_counts(
                _read_latest_state(pipeline.output_path, sample_name)
            )
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
            "start": math.floor(start),
            "end": math.floor(end),
            "llm": response_column_name,
            "dataset": llm_response_data["source"][i],
            "prompt": prompt,
            "claims": claims,
            "result": result_label,
        }
        if error_message is not None:
            result["error"] = error_message

        with open(os.path.join(dirpath, "eval_result.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle)