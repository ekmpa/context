from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import pandas as pd
from dotenv import load_dotenv

from llm_utils import query_qwen_batch
from model_utils import infer_backend_from_model, is_hf_local_backend

from analysis_prompt_utils import (
    SUPPORTED_CONDITIONS,
    condition_requires_web_search,
    judge_prompts,
    labels_for_condition,
    normalize_judge_label,
    summary_template,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DATASET_INFO_PATH = SCRIPTS_DIR / "dataset_info.json"


def _load_env():
    """Load environment variables from .env without overriding existing values."""
    env_file = ROOT_DIR / ".env"
    load_dotenv(dotenv_path=env_file, override=False)


# Load .env at module import time
_load_env()


def _normalize_dataset_name(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    return text.replace("-", "_").replace(" ", "_")


def _normalize_condition_name(value: str) -> str:
    return (value or "").strip().lower()


def _as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def parse_args() -> argparse.Namespace:
    condition_choices = sorted(set(SUPPORTED_CONDITIONS) | {"compare"})
    parser = argparse.ArgumentParser(description="Analyze claim datasets with condition-specific counting and LLM-as-judge modes.")
    parser.add_argument("--dataset", default="", help="Dataset name under data/<dataset> or a direct file path (.jsonl/.csv)")
    parser.add_argument(
        "--condition",
        required=True,
        choices=condition_choices,
        help="Analysis condition: none | conflict | conflict_compare | compare | stale | opinion | unverif | ambig",
    )
    parser.add_argument("--judge", default="gpt-4.1-mini", help="Judge model id")
    parser.add_argument("--output-dir", required=True, help="Directory to save analysis outputs")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on number of rows to analyze")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel judge requests for LLM conditions (default: 1)",
    )
    parser.add_argument(
        "--rarr-model",
        default="",
        help="Conflict mode filter: expected RARR model id in meta_eval.json (for example gpt-5-mini)",
    )
    parser.add_argument(
        "--rarr-condition",
        default="",
        help="Conflict mode filter: expected run condition in meta_eval.json (for example raw, structural, third-party)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Conflict mode filter: expected run seed in meta_eval.json",
    )
    parser.add_argument(
        "--run-log",
        default="",
        help="Path to a RARR model log (factcheck-*.json). Used by condition=conflict.",
    )
    parser.add_argument(
        "--compare-conditions",
        default="raw,structural,third-party,source_attr",
        help="Comma-separated RARR conditions to compare for condition=conflict_compare",
    )
    parser.add_argument(
        "--anchor-condition",
        default="raw",
        help="Condition used to define the conflicting-evidence subset for condition=conflict_compare",
    )
    parser.add_argument("--third-party-ratings-file", default="", help="Accepted for IO compatibility; not used yet")
    return parser.parse_args()


def _load_dataset_info() -> dict:
    """Load dataset_info.json; return empty dict if missing."""
    if DATASET_INFO_PATH.exists():
        try:
            return json.loads(DATASET_INFO_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _resolve_claim_dates(df: pd.DataFrame, dataset_key: str) -> list[str | None]:
    """Return a per-row date string (or None) for use in stale prompts.

    Resolution order:
    1. Per-row date from a dataset_info.json date_col (if non-null and parseable)
    2. A dataset-level dataset_year string from dataset_info.json
    3. Prompt the user interactively, then persist the answer to dataset_info.json
    """
    info = _load_dataset_info()
    key = dataset_key.strip().lower()
    entry: dict = info.get(key, {})

    date_col = entry.get("date_col")
    date_fmt = entry.get("date_col_format")
    dataset_year = entry.get("dataset_year")

    # --- Try per-row date column ---
    if date_col and date_col in df.columns:
        results: list[str | None] = []
        for val in df[date_col]:
            v = str(val).strip() if val is not None else ""
            if not v or v.lower() in {"na", "nan", "none", "null", ""}:
                results.append(str(dataset_year) if dataset_year else None)
            else:
                # Try to parse and reformat as "Month YYYY" for readability
                if date_fmt:
                    try:
                        from datetime import datetime as _dt
                        parsed = _dt.strptime(v, date_fmt)
                        results.append(parsed.strftime("%B %Y"))
                        continue
                    except ValueError:
                        pass
                results.append(v)
        return results

    # --- Fall back to dataset_year ---
    if dataset_year:
        print(f"[analyze_data] No per-row date for '{key}'; using dataset_year={dataset_year} for all rows.")
        return [str(dataset_year)] * len(df)

    # --- Unknown: ask the user and persist ---
    print(
        f"\n[analyze_data] Dataset '{key}' has no date information in scripts/dataset_info.json.\n"
        f"  What year were claims in this dataset collected? (e.g. 2018)\n"
        f"  Press Enter to skip (will use generic '12 months' phrasing): ",
        end="",
        flush=True,
    )
    answer = input().strip()
    if answer.isdigit():
        yr = int(answer)
        # Persist to dataset_info.json
        entry["dataset_year"] = yr
        if not entry.get("notes"):
            entry["notes"] = f"Year entered interactively by user."
        info[key] = entry
        # Remove comment key before writing to avoid JSON lint noise
        out_info = {k: v for k, v in info.items() if not k.startswith("_")}
        # Preserve _comment
        comment = info.get("_comment")
        if comment:
            out_info = {"_comment": comment, **out_info}
        DATASET_INFO_PATH.write_text(json.dumps(out_info, indent=2), encoding="utf-8")
        print(f"[analyze_data] Saved dataset_year={yr} for '{key}' to {DATASET_INFO_PATH}")
        return [str(yr)] * len(df)
    else:
        print("[analyze_data] No year provided; stale prompt will use generic '12 months' phrasing.")
        return [None] * len(df)


def _load_dataset_records(dataset_arg: str, max_rows: int | None) -> pd.DataFrame:
    dataset_path = Path(dataset_arg)
    if dataset_path.suffix.lower() in {".jsonl", ".csv"}:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        if dataset_path.suffix.lower() == ".jsonl":
            df = pd.read_json(dataset_path, lines=True)
        else:
            df = pd.read_csv(dataset_path)
    else:
        base_dir = ROOT_DIR / "data" / dataset_arg
        split_paths = [
            base_dir / "train" / "data.jsonl",
            base_dir / "val" / "data.jsonl",
            base_dir / "test" / "data.jsonl",
        ]
        found = [path for path in split_paths if path.exists()]
        if not found:
            raise FileNotFoundError(
                f"No split files found for dataset '{dataset_arg}'. Expected one of: "
                + ", ".join(str(p) for p in split_paths)
            )
        frames = [pd.read_json(path, lines=True) for path in found]
        df = pd.concat(frames, ignore_index=True)

    if "claim" in df.columns:
        claim_col = "claim"
    elif "prompt" in df.columns:
        claim_col = "prompt"
    else:
        raise ValueError("Dataset must contain either a 'claim' column or a 'prompt' column")

    if "label" not in df.columns:
        df["label"] = "<missing>"

    out = df[[claim_col, "label"]].copy()
    out = out.rename(columns={claim_col: "claim"})
    out["claim"] = out["claim"].astype(str).str.strip()
    out["label"] = out["label"].astype(str).str.strip()
    out = out[out["claim"].astype(bool)].reset_index(drop=True)

    if max_rows is not None:
        out = out.head(max_rows).reset_index(drop=True)

    return out


def _infer_backend_from_model(model_name: str) -> str:
    return infer_backend_from_model(model_name)


def _is_hf_local_backend() -> bool:
    backend = os.getenv("CONTEXT_LLM_BACKEND", "openai")
    return is_hf_local_backend(backend)


def _format_hf_local_prompt(system_prompt: str, user_prompt: str) -> str:
    return f"SYSTEM: {system_prompt.strip()}\nUSER: {user_prompt.strip()}"


def _query_no_web(model: str, system_prompt: str, user_prompt: str, use_web_search: bool) -> str:
    if use_web_search:
        raise RuntimeError("This backend does not support web_search-enabled judging")

    from context_core.llm import chat_text

    return chat_text(
        [{"role": "user", "content": user_prompt}],
        model=model,
        system_role=system_prompt,
        temperature=0.0,
    )


def _query_openai_responses(model: str, system_prompt: str, user_prompt: str, use_web_search: bool) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI judge calls require the openai package") from exc

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI judge calls")

    client = OpenAI(api_key=api_key)
    request_kwargs: dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": [{"role": "user", "content": user_prompt}],
    }

    if use_web_search:
        request_kwargs["tools"] = [{"type": "web_search"}]
        request_kwargs["tool_choice"] = "required"

    response = client.responses.create(**request_kwargs)
    return str(getattr(response, "output_text", "")).strip()


def _condition_none(df: pd.DataFrame) -> dict[str, Any]:
    counts = Counter(df["label"].tolist())
    summary = summary_template("custom", "none", len(df))
    summary["label_counts"] = dict(sorted(counts.items(), key=lambda kv: kv[0]))
    return summary


def _condition_conflict_placeholder(df: pd.DataFrame) -> dict[str, Any]:
    summary = summary_template("custom", "conflict", len(df))
    summary["placeholder"] = True
    summary["message"] = "conflict analysis placeholder; implementation pending"
    return summary


def _resolve_conflict_run_log(
    explicit_path: str,
    *,
    dataset: str,
    rarr_model: str,
    rarr_condition: str,
    required_structural_hops: int | None = None,
    seed: int | None,
) -> Path:
    if explicit_path.strip():
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Run log not found: {path}")
        return path

    dataset_filter = _normalize_dataset_name(dataset)
    model_filter = rarr_model.strip().lower()
    # Conflict analysis is intended for fact-checking evidence behavior from the
    # standard raw pipeline unless explicitly overridden.
    condition_filter = _normalize_condition_name(rarr_condition or "raw")

    # Prefer explicitly matching metadata, then recency, then run size.
    strict_candidates: list[tuple[int, int, int, float, int, Path]] = []
    fallback_candidates: list[tuple[int, int, int, float, int, Path]] = []
    for meta_path in (ROOT_DIR / "eval_results" / "custom").glob("rarr-*/meta_eval.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta_dataset = _normalize_dataset_name(str(meta.get("dataset", "") or ""))
        meta_model = str(((meta.get("model") or {}).get("rarr_model", "") or "")).strip().lower()
        meta_condition = _normalize_condition_name(str(meta.get("condition", "") or ""))
        meta_seed = meta.get("seed", None)
        rows_total = int(meta.get("rows_total", 0) or 0)
        structural_hops = _as_int(meta.get("structural_hops"))

        if required_structural_hops is not None:
            if meta_condition != "structural":
                continue
            if structural_hops != required_structural_hops:
                continue

        dataset_quality = 1
        if dataset_filter:
            if meta_dataset:
                if meta_dataset != dataset_filter:
                    continue
                dataset_quality = 2
            else:
                dataset_quality = 0

        model_quality = 1
        if model_filter:
            if meta_model:
                if meta_model != model_filter:
                    continue
                model_quality = 2
            else:
                model_quality = 0

        if condition_filter and meta_condition and meta_condition != condition_filter:
            continue

        condition_quality = 1 if not condition_filter else (2 if meta_condition else 0)

        # For structural condition, prefer the temporal 1-hop / 2-hop runs.
        structural_hops_quality = 1
        if condition_filter == "structural":
            if structural_hops in {1, 2}:
                structural_hops_quality = 3
            elif structural_hops is not None:
                structural_hops_quality = 2
            else:
                structural_hops_quality = 0

        if seed is not None:
            if isinstance(meta_seed, int):
                if meta_seed != seed:
                    continue
            else:
                # If seed is requested but not recorded, skip to avoid accidental mismatches.
                continue

        run_dir = meta_path.parent
        model_logs = sorted(run_dir.glob("factcheck-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not model_logs:
            continue

        newest_log = model_logs[0]
        unknown_name_penalty = 0 if "unknown" not in newest_log.name.lower() else -1
        candidate = (
            dataset_quality,
            condition_quality,
            structural_hops_quality,
            model_quality,
            meta_path.stat().st_mtime,
            rows_total,
            newest_log.resolve(),
        )

        if dataset_quality == 2 and condition_quality == 2 and (model_quality > 0):
            strict_candidates.append(candidate)
        else:
            # Keep weaker metadata matches as a fallback, while still preferring
            # non-unknown file names and more complete runs within the tier.
            fallback_candidates.append(
                (
                    dataset_quality,
                    condition_quality,
                    structural_hops_quality,
                    model_quality + unknown_name_penalty,
                    meta_path.stat().st_mtime,
                    rows_total,
                    newest_log.resolve(),
                )
            )

    strict_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]), reverse=True)
    fallback_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]), reverse=True)
    resolved = [path for _, _, _, _, _, _, path in strict_candidates]
    resolved.extend(path for _, _, _, _, _, _, path in fallback_candidates)
    if not resolved:
        raise FileNotFoundError(
            "No matching run log found under eval_results/custom for the requested filters. "
            "Provide --run-log <path/to/factcheck-*.json> or relax dataset/model/condition/seed filters."
        )
    return resolved[0]


def _list_available_run_seeds(
    *,
    dataset: str,
    rarr_model: str,
    rarr_condition: str,
    required_structural_hops: int | None = None,
) -> list[int]:
    dataset_filter = _normalize_dataset_name(dataset)
    model_filter = (rarr_model or "").strip().lower()
    condition_filter = _normalize_condition_name(rarr_condition)

    seeds: set[int] = set()
    for meta_path in (ROOT_DIR / "eval_results" / "custom").glob("rarr-*/meta_eval.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta_dataset = _normalize_dataset_name(str(meta.get("dataset", "") or ""))
        meta_condition = _normalize_condition_name(str(meta.get("condition", "") or ""))
        meta_model = str(((meta.get("model") or {}).get("rarr_model", "") or "")).strip().lower()
        meta_seed = _as_int(meta.get("seed"))
        structural_hops = _as_int(meta.get("structural_hops"))

        if dataset_filter and meta_dataset and meta_dataset != dataset_filter:
            continue
        if condition_filter and meta_condition and meta_condition != condition_filter:
            continue
        if required_structural_hops is not None:
            if meta_condition != "structural" or structural_hops != required_structural_hops:
                continue
        if model_filter and meta_model and meta_model != model_filter:
            continue
        if meta_seed is not None:
            seeds.add(meta_seed)

    return sorted(seeds)


def _mean_std(values: list[float | int]) -> tuple[float | None, float | None]:
    numeric = [float(v) for v in values]
    if not numeric:
        return None, None
    return mean(numeric), (pstdev(numeric) if len(numeric) > 1 else 0.0)


def _condition_conflict_from_run_log(run_log_path: Path, max_rows: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(run_log_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Unexpected run log format in {run_log_path}: 'entries' must be a list")

    if max_rows is not None:
        entries = entries[:max_rows]

    rows: list[dict[str, Any]] = []
    conflict_count = 0
    directional_count = 0
    with_labels_count = 0
    pattern_counts: Counter[str] = Counter()

    for idx, entry in enumerate(entries):
        details = entry.get("detail", []) if isinstance(entry, dict) else []
        if not isinstance(details, list):
            details = []

        labels: list[str] = []
        claim_text = str(entry.get("prompt", "") or "").strip()

        for detail in details:
            if not isinstance(detail, dict):
                continue
            if not claim_text:
                claim_text = str(detail.get("claim", "") or "").strip()
            detail_labels = detail.get("labels", [])
            if isinstance(detail_labels, list):
                labels.extend(str(label).strip().lower() for label in detail_labels if str(label).strip())

        agrees = sum(1 for label in labels if label == "agrees")
        disagrees = sum(1 for label in labels if label == "disagrees")
        ambiguous = sum(1 for label in labels if label == "ambiguous")
        unverifiable = sum(1 for label in labels if label == "unverifiable")

        has_labels = bool(labels)
        has_directional = (agrees + disagrees) > 0
        has_conflict = agrees > 0 and disagrees > 0

        if has_labels:
            with_labels_count += 1
        if has_directional:
            directional_count += 1
        if has_conflict:
            conflict_count += 1

        if has_conflict:
            pattern = "mixed_support_and_refute"
        elif agrees > 0 and disagrees == 0:
            pattern = "support_only"
        elif disagrees > 0 and agrees == 0:
            pattern = "refute_only"
        elif ambiguous > 0:
            pattern = "ambiguous_only"
        elif unverifiable > 0:
            pattern = "unverifiable_only"
        else:
            pattern = "no_labels"
        pattern_counts[pattern] += 1

        rows.append(
            {
                "index": idx,
                "claim": claim_text,
                "total_labels": len(labels),
                "counts": {
                    "agrees": agrees,
                    "disagrees": disagrees,
                    "ambiguous": ambiguous,
                    "unverifiable": unverifiable,
                },
                "has_directional_signal": has_directional,
                "has_conflicting_signal": has_conflict,
                "pattern": pattern,
            }
        )

    total = len(entries)
    summary = summary_template("custom", "conflict", total)
    summary["run_log"] = str(run_log_path)
    summary["rows_with_labels"] = with_labels_count
    summary["rows_with_directional_signal"] = directional_count
    summary["rows_with_conflicting_signal"] = conflict_count
    summary["conflict_rate_of_total"] = (conflict_count / total) if total else 0.0
    summary["conflict_rate_of_directional"] = (conflict_count / directional_count) if directional_count else 0.0
    summary["pattern_counts"] = dict(sorted(pattern_counts.items(), key=lambda kv: kv[0]))
    return summary, rows


def _extract_entry_labels(entry: dict[str, Any]) -> list[str]:
    details = entry.get("detail", []) if isinstance(entry, dict) else []
    if not isinstance(details, list):
        return []
    labels: list[str] = []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        detail_labels = detail.get("labels", [])
        if isinstance(detail_labels, list):
            labels.extend(str(label).strip().lower() for label in detail_labels if str(label).strip())
    return labels


def _entry_has_conflicting_signal(entry: dict[str, Any]) -> bool:
    labels = _extract_entry_labels(entry)
    return ("agrees" in labels) and ("disagrees" in labels)


def _entry_key(entry: dict[str, Any]) -> tuple[str, str | int] | None:
    raw_index = entry.get("index")
    if isinstance(raw_index, int):
        return ("index", raw_index)
    if isinstance(raw_index, str) and raw_index.strip().isdigit():
        return ("index", int(raw_index.strip()))
    claim = str(entry.get("prompt", "") or "").strip().lower()
    if claim:
        return ("claim", claim)
    return None


def _load_run_entries(run_log_path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    payload = json.loads(run_log_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Unexpected run log format in {run_log_path}: 'entries' must be a list")
    if max_rows is not None:
        entries = entries[:max_rows]
    cleaned: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict):
            cleaned.append(entry)
    return cleaned


def _parse_compare_conditions(text: str) -> list[str]:
    valid_conditions = {"raw", "structural", "third-party", "source_attr", "struct-1t", "struct-2t"}
    out: list[str] = []
    for chunk in text.split(","):
        cond = _normalize_condition_name(chunk)
        if cond and cond not in valid_conditions:
            raise ValueError(
                "Unsupported compare condition "
                f"'{cond}'. Use one of: raw, structural, third-party, source_attr, struct-1t, struct-2t"
            )
        if cond and cond not in out:
            out.append(cond)
    return out


def _expand_compare_condition_specs(compare_conditions: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    def _push(label: str, run_condition: str, structural_hops: int | None = None) -> None:
        if label in seen_labels:
            return
        seen_labels.add(label)
        specs.append(
            {
                "label": label,
                "run_condition": run_condition,
                "structural_hops": structural_hops,
            }
        )

    for cond in compare_conditions:
        if cond == "structural":
            _push("struct-1t", "structural", structural_hops=1)
            _push("struct-2t", "structural", structural_hops=2)
            continue
        if cond == "struct-1t":
            _push("struct-1t", "structural", structural_hops=1)
            continue
        if cond == "struct-2t":
            _push("struct-2t", "structural", structural_hops=2)
            continue
        _push(cond, cond)

    return specs


def _select_runs_for_compare(
    *,
    dataset: str,
    rarr_model: str,
    seed: int | None,
    condition_specs: list[dict[str, Any]],
    max_rows: int | None,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    display_conditions = [spec["label"] for spec in condition_specs]

    selected: dict[str, dict[str, Any]] = {}
    missing_conditions: list[str] = []
    for spec in condition_specs:
        cond = str(spec["label"])
        seed_values = [seed] if seed is not None else _list_available_run_seeds(
            dataset=dataset,
            rarr_model=rarr_model,
            rarr_condition=str(spec["run_condition"]),
            required_structural_hops=spec.get("structural_hops"),
        )
        if not seed_values:
            seed_values = [None]

        runs_by_seed: dict[str, dict[str, Any]] = {}
        representative_payload: dict[str, Any] | None = None
        for seed_value in seed_values:
            try:
                run_log_path = _resolve_conflict_run_log(
                    "",
                    dataset=dataset,
                    rarr_model=rarr_model,
                    rarr_condition=str(spec["run_condition"]),
                    required_structural_hops=spec.get("structural_hops"),
                    seed=seed_value,
                )
            except FileNotFoundError:
                continue

            entries = _load_run_entries(run_log_path, max_rows)
            key_map: dict[tuple[str, str | int], dict[str, Any]] = {}
            for entry in entries:
                key = _entry_key(entry)
                if key is not None:
                    key_map[key] = entry

            meta_path = run_log_path.parent / "meta_eval.json"
            meta: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}

            seed_key = str(seed_value) if seed_value is not None else "unseeded"
            payload = {
                "run_log_path": run_log_path,
                "entries": entries,
                "by_key": key_map,
                "meta": meta,
                "seed": seed_value,
            }
            runs_by_seed[seed_key] = payload
            if representative_payload is None:
                representative_payload = payload

        if not runs_by_seed:
            missing_conditions.append(cond)
            continue

        selected[cond] = {
            "runs_by_seed": runs_by_seed,
            "representative": representative_payload,
        }

    return selected, display_conditions, missing_conditions


def _resolve_anchor_condition(
    *,
    selected: dict[str, dict[str, Any]],
    anchor_condition: str,
    report_name: str,
) -> str:
    if anchor_condition in selected:
        return anchor_condition

    if selected:
        return next(iter(selected.keys()))

    raise FileNotFoundError(
        f"No matching runs found for {report_name} under eval_results/custom. "
        "Run eval first, or relax --dataset/--rarr-model/--seed filters."
    )


def _collect_subset_keys(
    entries: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> list[tuple[str, str | int]]:
    keys: list[tuple[str, str | int]] = []
    for entry in entries:
        key = _entry_key(entry)
        if key is not None and predicate(entry):
            keys.append(key)
    return list(dict.fromkeys(keys))


def _compute_per_condition_subset_stats(
    *,
    selected: dict[str, dict[str, Any]],
    display_conditions: list[str],
    anchor_condition: str,
    subset_predicate: Callable[[dict[str, Any]], bool],
    subset_label: str,
) -> dict[str, Any]:
    rows_field = f"rows_on_{subset_label}_subset"
    correct_field = f"correct_on_{subset_label}_subset"
    acc_field = f"accuracy_on_{subset_label}_subset"

    per_condition: dict[str, Any] = {}
    anchor_runs_by_seed = selected[anchor_condition]["runs_by_seed"]
    for cond in display_conditions:
        payload = selected.get(cond)
        if not payload:
            continue
        runs_by_seed = payload["runs_by_seed"]
        common_seed_keys = sorted(set(anchor_runs_by_seed) & set(runs_by_seed), key=str)
        per_seed_metrics: list[dict[str, Any]] = []
        for seed_key in common_seed_keys:
            seed_anchor = anchor_runs_by_seed[seed_key]
            seed_target = runs_by_seed[seed_key]
            seed_subset_keys = _collect_subset_keys(seed_anchor["entries"], subset_predicate)
            by_key = seed_target["by_key"]
            evaluated_keys = [k for k in seed_subset_keys if k in by_key]
            total = len(evaluated_keys)
            correct = sum(1 for k in evaluated_keys if bool(by_key[k].get("result", False)))
            accuracy = (correct / total) if total else None
            overall_accuracy = (seed_target.get("meta") or {}).get("accuracy")
            per_seed_metrics.append(
                {
                    "seed": seed_target.get("seed"),
                    "seed_key": seed_key,
                    rows_field: total,
                    correct_field: correct,
                    acc_field: accuracy,
                    "overall_accuracy": overall_accuracy,
                }
            )

        rows_mean, rows_std = _mean_std([m[rows_field] for m in per_seed_metrics])
        correct_mean, correct_std = _mean_std([m[correct_field] for m in per_seed_metrics])
        acc_mean, acc_std = _mean_std([
            m[acc_field] for m in per_seed_metrics if isinstance(m[acc_field], (int, float))
        ])
        overall_mean, overall_std = _mean_std([
            m["overall_accuracy"] for m in per_seed_metrics if isinstance(m["overall_accuracy"], (int, float))
        ])
        rep = payload["representative"]
        per_condition[cond] = {
            "run_log": str(rep["run_log_path"]),
            "rows_total": len(rep["entries"]),
            "seed_count": len(per_seed_metrics),
            "seed_keys": common_seed_keys,
            "per_seed": per_seed_metrics,
            rows_field: rows_mean,
            f"{rows_field}_std": rows_std,
            correct_field: correct_mean,
            f"{correct_field}_std": correct_std,
            acc_field: acc_mean,
            f"{acc_field}_std": acc_std,
            "overall_accuracy": overall_mean,
            "overall_accuracy_std": overall_std,
            "run_id": (rep.get("meta") or {}).get("run_id"),
        }

    anchor_stats = per_condition.get(anchor_condition)
    anchor_subset_acc = anchor_stats.get(acc_field) if isinstance(anchor_stats, dict) else None
    anchor_overall_acc = anchor_stats.get("overall_accuracy") if isinstance(anchor_stats, dict) else None

    for cond in display_conditions:
        stats = per_condition.get(cond)
        if not isinstance(stats, dict):
            continue
        subset_acc = stats.get(acc_field)
        overall_acc = stats.get("overall_accuracy")
        stats["gain_vs_anchor_subset_pct"] = (
            ((subset_acc - anchor_subset_acc) / anchor_subset_acc) * 100.0
            if (
                isinstance(subset_acc, (int, float))
                and isinstance(anchor_subset_acc, (int, float))
                and anchor_subset_acc != 0
            )
            else None
        )
        stats["gain_vs_anchor_overall_pct"] = (
            ((overall_acc - anchor_overall_acc) / anchor_overall_acc) * 100.0
            if (
                isinstance(overall_acc, (int, float))
                and isinstance(anchor_overall_acc, (int, float))
                and anchor_overall_acc != 0
            )
            else None
        )

    return per_condition


def _build_subset_compare_rows(
    *,
    selected: dict[str, dict[str, Any]],
    display_conditions: list[str],
    anchor_condition: str,
    subset_keys: list[tuple[str, str | int]],
    subset_predicate: Callable[[dict[str, Any]], bool],
    anchor_flag_name: str,
) -> list[dict[str, Any]]:
    anchor_by_key = selected[anchor_condition]["representative"]["by_key"]
    rows: list[dict[str, Any]] = []
    for key in subset_keys:
        anchor_entry = anchor_by_key.get(key, {})
        row: dict[str, Any] = {
            "key_type": key[0],
            "key": key[1],
            "claim": str(anchor_entry.get("prompt", "") or "").strip(),
            "gold_label": str(anchor_entry.get("gold_label", "") or "").strip().lower(),
            "anchor_condition": anchor_condition,
            anchor_flag_name: subset_predicate(anchor_entry),
            "per_condition": {},
        }
        for cond in display_conditions:
            payload = selected.get(cond)
            if not payload:
                row["per_condition"][cond] = None
                continue
            representative = payload.get("representative") or {}
            entry = representative.get("by_key", {}).get(key)
            if entry is None:
                row["per_condition"][cond] = None
                continue
            row["per_condition"][cond] = {
                "pred_label": str(entry.get("pred_label", "") or "").strip().lower(),
                "pred_bucket": str(entry.get("pred_bucket", "") or "").strip().lower(),
                "correct": bool(entry.get("result", False)),
                "has_conflicting_signal": _entry_has_conflicting_signal(entry),
            }
        rows.append(row)

    return rows


def _condition_subset_compare(
    *,
    dataset: str,
    rarr_model: str,
    seed: int | None,
    compare_conditions: list[str],
    anchor_condition: str,
    max_rows: int | None,
    report_name: str,
    subset_label: str,
    subset_predicate: Callable[[dict[str, Any]], bool],
    anchor_flag_name: str,
    summary_subset_count_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not compare_conditions:
        raise ValueError(f"No compare conditions provided for {report_name}")

    condition_specs = _expand_compare_condition_specs(compare_conditions)
    selected, display_conditions, missing_conditions = _select_runs_for_compare(
        dataset=dataset,
        rarr_model=rarr_model,
        seed=seed,
        condition_specs=condition_specs,
        max_rows=max_rows,
    )

    anchor_condition = _resolve_anchor_condition(
        selected=selected,
        anchor_condition=anchor_condition,
        report_name=report_name,
    )

    anchor_rep = selected[anchor_condition]["representative"]
    subset_keys = _collect_subset_keys(anchor_rep["entries"], subset_predicate)
    per_condition = _compute_per_condition_subset_stats(
        selected=selected,
        display_conditions=display_conditions,
        anchor_condition=anchor_condition,
        subset_predicate=subset_predicate,
        subset_label=subset_label,
    )
    rows = _build_subset_compare_rows(
        selected=selected,
        display_conditions=display_conditions,
        anchor_condition=anchor_condition,
        subset_keys=subset_keys,
        subset_predicate=subset_predicate,
        anchor_flag_name=anchor_flag_name,
    )

    summary = summary_template(dataset or "custom", report_name, len(subset_keys))
    summary["anchor_condition"] = anchor_condition
    summary["anchor_run_log"] = str(anchor_rep["run_log_path"])
    summary[summary_subset_count_key] = len(subset_keys)
    summary["compare_conditions"] = display_conditions
    summary["missing_conditions"] = missing_conditions
    summary["per_condition"] = per_condition
    return summary, rows


def _condition_conflict_compare(
    *,
    dataset: str,
    rarr_model: str,
    seed: int | None,
    compare_conditions: list[str],
    anchor_condition: str,
    max_rows: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _condition_subset_compare(
        dataset=dataset,
        rarr_model=rarr_model,
        seed=seed,
        compare_conditions=compare_conditions,
        anchor_condition=anchor_condition,
        max_rows=max_rows,
        report_name="conflict_compare",
        subset_label="conflict",
        subset_predicate=_entry_has_conflicting_signal,
        anchor_flag_name="anchor_has_conflicting_signal",
        summary_subset_count_key="conflict_rows_anchor",
    )


def _entry_is_unverifiable_by_gold_label(entry: dict[str, Any]) -> bool:
    gold = str(entry.get("gold_label", "") or "").strip().lower()
    return gold in {"unverified", "unverifiable"}


def _condition_unverifiable_compare(
    *,
    dataset: str,
    rarr_model: str,
    seed: int | None,
    compare_conditions: list[str],
    anchor_condition: str,
    max_rows: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    return _condition_subset_compare(
        dataset=dataset,
        rarr_model=rarr_model,
        seed=seed,
        compare_conditions=compare_conditions,
        anchor_condition=anchor_condition,
        max_rows=max_rows,
        report_name="unverifiable_compare",
        subset_label="unverifiable",
        subset_predicate=_entry_is_unverifiable_by_gold_label,
        anchor_flag_name="anchor_is_unverifiable",
        summary_subset_count_key="unverifiable_rows_anchor",
    )


def _resolve_compare_cli_inputs(compare_conditions_arg: str, anchor_condition_arg: str) -> tuple[list[str], str]:
    compare_conditions = _parse_compare_conditions(compare_conditions_arg)
    if not compare_conditions:
        raise ValueError("--compare-conditions must include at least one condition")

    anchor_parsed = _parse_compare_conditions(anchor_condition_arg.strip())
    anchor_condition = anchor_parsed[0] if anchor_parsed else ""
    if anchor_condition and anchor_condition not in compare_conditions:
        compare_conditions = [anchor_condition, *compare_conditions]
    return compare_conditions, (anchor_condition or "raw")


def _run_llm_condition(
    *,
    df: pd.DataFrame,
    condition: str,
    model: str,
    query_fn: Callable[[str, str, str, bool], str],
    use_web_search: bool,
    workers: int,
    claim_dates: list[str | None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = labels_for_condition(condition)
    label_counts: Counter[str] = Counter({k: 0 for k in labels})
    rows: list[dict[str, Any] | None] = [None] * len(df)

    total = len(df)
    resolved_workers = max(1, int(workers))
    local_backend = _is_hf_local_backend()
    print(
        f"[analyze_data] Running condition='{condition}' on {total} rows with judge='{model}' "
        f"(workers={resolved_workers})"
    )

    claims = df["claim"].tolist()
    processed = 0
    if local_backend:
        request_payloads: list[tuple[int, str, str | None, str]] = []
        for idx, claim in enumerate(claims):
            cdate = claim_dates[idx] if claim_dates and idx < len(claim_dates) else None
            system_prompt, user_prompt = judge_prompts(condition, claim, claim_date=cdate)
            request_payloads.append((idx, claim, cdate, _format_hf_local_prompt(system_prompt, user_prompt)))

        for start in range(0, total, resolved_workers):
            batch = request_payloads[start : start + resolved_workers]
            raw_outputs = query_qwen_batch(model, [item[3] for item in batch])
            if len(raw_outputs) != len(batch):
                raise RuntimeError(
                    f"Unexpected local judge output count: got {len(raw_outputs)}, expected {len(batch)}"
                )
            for (idx, claim, cdate, _), raw in zip(batch, raw_outputs):
                norm = normalize_judge_label(condition, raw)
                row: dict[str, Any] = {
                    "index": idx,
                    "claim": claim,
                    "judge_label": norm,
                    "judge_raw": raw,
                }
                if cdate is not None:
                    row["claim_date"] = cdate
                label_counts[row["judge_label"]] += 1
                rows[idx] = row
                processed += 1
            if processed % 25 == 0 or processed == total:
                print(f"[analyze_data]   processed {processed}/{total}")
    else:
        def _process_one(idx: int, claim: str) -> tuple[int, dict[str, Any]]:
            cdate = claim_dates[idx] if claim_dates and idx < len(claim_dates) else None
            system_prompt, user_prompt = judge_prompts(condition, claim, claim_date=cdate)
            raw = query_fn(model, system_prompt, user_prompt, use_web_search)
            norm = normalize_judge_label(condition, raw)
            row: dict[str, Any] = {
                "index": idx,
                "claim": claim,
                "judge_label": norm,
                "judge_raw": raw,
            }
            if cdate is not None:
                row["claim_date"] = cdate
            return idx, row

        with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_workers) as pool:
            futures = [pool.submit(_process_one, idx, claim) for idx, claim in enumerate(claims)]
            for fut in concurrent.futures.as_completed(futures):
                idx, row = fut.result()
                label_counts[row["judge_label"]] += 1
                rows[idx] = row
                processed += 1
                if processed % 25 == 0 or processed == total:
                    print(f"[analyze_data]   processed {processed}/{total}")

    finalized_rows = [row for row in rows if row is not None]

    summary = summary_template("custom", condition, total)
    summary["judge_model"] = model
    summary["workers"] = resolved_workers
    summary["backend"] = "hf-local" if local_backend else "openai"
    summary["counts"] = dict(sorted(label_counts.items(), key=lambda kv: kv[0]))
    return summary, finalized_rows


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Analysis Summary ===")
    print(f"condition: {summary.get('condition')}")
    print(f"total: {summary.get('total')}")

    if "label_counts" in summary:
        print("label counts:")
        for key, value in summary["label_counts"].items():
            print(f"  {key}: {value}")
        return

    if "counts" in summary:
        print("judge counts:")
        for key, value in summary["counts"].items():
            print(f"  {key}: {value}")
        return

    if "rows_with_conflicting_signal" in summary:
        print(f"run_log: {summary.get('run_log')}")
        print(f"rows_with_labels: {summary.get('rows_with_labels')}")
        print(f"rows_with_directional_signal: {summary.get('rows_with_directional_signal')}")
        print(f"rows_with_conflicting_signal: {summary.get('rows_with_conflicting_signal')}")
        print(f"conflict_rate_of_total: {summary.get('conflict_rate_of_total')}")
        print(f"conflict_rate_of_directional: {summary.get('conflict_rate_of_directional')}")
        patterns = summary.get("pattern_counts", {})
        if isinstance(patterns, dict):
            print("pattern_counts:")
            for key, value in patterns.items():
                print(f"  {key}: {value}")
        return

    if summary.get("condition") == "conflict_compare":
        print(f"anchor_condition: {summary.get('anchor_condition')}")
        print(f"anchor_run_log: {summary.get('anchor_run_log')}")
        print(f"conflict_rows_anchor: {summary.get('conflict_rows_anchor')}")
        missing = summary.get("missing_conditions", [])
        if isinstance(missing, list) and missing:
            print(f"missing_conditions: {', '.join(str(x) for x in missing)}")
        per_condition = summary.get("per_condition", {})
        if isinstance(per_condition, dict):
            print("per-condition on conflict subset:")
            for cond in summary.get("compare_conditions", []):
                stats = per_condition.get(cond)
                if not isinstance(stats, dict):
                    print(f"  {cond}: <no matching run>")
                    continue
                print(
                    f"  {cond}: n_seeds={stats.get('seed_count', 0)}, "
                    f"rows={stats.get('rows_on_conflict_subset', 0)}, rows_std={stats.get('rows_on_conflict_subset_std')}, "
                    f"correct={stats.get('correct_on_conflict_subset', 0)}, correct_std={stats.get('correct_on_conflict_subset_std')}, "
                    f"acc={stats.get('accuracy_on_conflict_subset')}, "
                    f"acc_std={stats.get('accuracy_on_conflict_subset_std')}, "
                    f"overall_acc={stats.get('overall_accuracy')}, "
                    f"overall_acc_std={stats.get('overall_accuracy_std')}, "
                    f"gain_pct={stats.get('gain_vs_anchor_subset_pct')}, "
                    f"overall_gain_pct={stats.get('gain_vs_anchor_overall_pct')}"
                )
        return

    if summary.get("condition") == "unverifiable_compare":
        print(f"anchor_condition: {summary.get('anchor_condition')}")
        print(f"anchor_run_log: {summary.get('anchor_run_log')}")
        print(f"unverifiable_rows_anchor: {summary.get('unverifiable_rows_anchor')}")
        missing = summary.get("missing_conditions", [])
        if isinstance(missing, list) and missing:
            print(f"missing_conditions: {', '.join(str(x) for x in missing)}")
        per_condition = summary.get("per_condition", {})
        if isinstance(per_condition, dict):
            print("per-condition on unverified-label subset:")
            for cond in summary.get("compare_conditions", []):
                stats = per_condition.get(cond)
                if not isinstance(stats, dict):
                    print(f"  {cond}: <no matching run>")
                    continue
                print(
                    f"  {cond}: n_seeds={stats.get('seed_count', 0)}, "
                    f"rows={stats.get('rows_on_unverifiable_subset', 0)}, rows_std={stats.get('rows_on_unverifiable_subset_std')}, "
                    f"correct={stats.get('correct_on_unverifiable_subset', 0)}, correct_std={stats.get('correct_on_unverifiable_subset_std')}, "
                    f"acc={stats.get('accuracy_on_unverifiable_subset')}, "
                    f"acc_std={stats.get('accuracy_on_unverifiable_subset_std')}, "
                    f"overall_acc={stats.get('overall_accuracy')}, "
                    f"overall_acc_std={stats.get('overall_accuracy_std')}, "
                    f"gain_pct={stats.get('gain_vs_anchor_subset_pct')}, "
                    f"overall_gain_pct={stats.get('gain_vs_anchor_overall_pct')}"
                )
        return

    if summary.get("placeholder"):
        print(summary.get("message", "placeholder"))


def main() -> None:
    args = parse_args()
    print(f"[analyze_data] dataset={args.dataset}, condition={args.condition}, max_rows={args.max_rows}, workers={args.workers}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analyze_data] output_dir={output_dir}")

    if args.condition == "conflict":
        run_log_path = _resolve_conflict_run_log(
            args.run_log,
            dataset=args.dataset,
            rarr_model=args.rarr_model,
            rarr_condition=args.rarr_condition,
            seed=args.seed,
        )
        print(f"[analyze_data] conflict run_log={run_log_path}")
        summary, rows = _condition_conflict_from_run_log(run_log_path, args.max_rows)
        summary["dataset"] = args.dataset or "from_run_log"
        _print_summary(summary)

        summary_path = output_dir / "summary.json"
        rows_path = output_dir / "per_claim_conflict.jsonl"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        with open(rows_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        print(f"[analyze_data] Saved summary: {summary_path}")
        print(f"[analyze_data] Saved per-claim conflict: {rows_path}")
        return

    if args.condition == "conflict_compare":
        compare_conditions, anchor_condition = _resolve_compare_cli_inputs(
            args.compare_conditions,
            args.anchor_condition,
        )
        summary, rows = _condition_conflict_compare(
            dataset=args.dataset,
            rarr_model=args.rarr_model,
            seed=args.seed,
            compare_conditions=compare_conditions,
            anchor_condition=anchor_condition,
            max_rows=args.max_rows,
        )
        _print_summary(summary)

        summary_path = output_dir / "summary.json"
        rows_path = output_dir / "per_claim_conflict_compare.jsonl"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        with open(rows_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        print(f"[analyze_data] Saved summary: {summary_path}")
        print(f"[analyze_data] Saved per-claim conflict-compare: {rows_path}")
        return

    if args.condition == "compare":
        compare_conditions, anchor_condition = _resolve_compare_cli_inputs(
            args.compare_conditions,
            args.anchor_condition,
        )

        conflict_summary, conflict_rows = _condition_conflict_compare(
            dataset=args.dataset,
            rarr_model=args.rarr_model,
            seed=args.seed,
            compare_conditions=compare_conditions,
            anchor_condition=anchor_condition,
            max_rows=args.max_rows,
        )
        unverif_summary, unverif_rows = _condition_unverifiable_compare(
            dataset=args.dataset,
            rarr_model=args.rarr_model,
            seed=args.seed,
            compare_conditions=compare_conditions,
            anchor_condition=anchor_condition,
            max_rows=args.max_rows,
        )

        print("\n=== Compare Report: Conflict Subset ===")
        _print_summary(conflict_summary)
        print("\n=== Compare Report: Unverified-Label Subset ===")
        _print_summary(unverif_summary)

        conflict_summary_path = output_dir / "summary_conflict_compare.json"
        conflict_rows_path = output_dir / "per_claim_conflict_compare.jsonl"
        unverif_summary_path = output_dir / "summary_unverifiable_compare.json"
        unverif_rows_path = output_dir / "per_claim_unverifiable_compare.jsonl"
        bundled_summary_path = output_dir / "summary.json"

        with open(conflict_summary_path, "w", encoding="utf-8") as handle:
            json.dump(conflict_summary, handle, indent=2, sort_keys=True)
        with open(conflict_rows_path, "w", encoding="utf-8") as handle:
            for row in conflict_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        with open(unverif_summary_path, "w", encoding="utf-8") as handle:
            json.dump(unverif_summary, handle, indent=2, sort_keys=True)
        with open(unverif_rows_path, "w", encoding="utf-8") as handle:
            for row in unverif_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        bundled_summary = {
            "dataset": args.dataset,
            "condition": "compare",
            "compare_conditions": compare_conditions,
            "anchor_condition": anchor_condition,
            "reports": {
                "conflict_compare": conflict_summary,
                "unverifiable_compare": unverif_summary,
            },
        }
        with open(bundled_summary_path, "w", encoding="utf-8") as handle:
            json.dump(bundled_summary, handle, indent=2, sort_keys=True)

        print(f"[analyze_data] Saved conflict summary: {conflict_summary_path}")
        print(f"[analyze_data] Saved conflict per-claim: {conflict_rows_path}")
        print(f"[analyze_data] Saved unverifiable summary: {unverif_summary_path}")
        print(f"[analyze_data] Saved unverifiable per-claim: {unverif_rows_path}")
        print(f"[analyze_data] Saved bundled summary: {bundled_summary_path}")
        return

    if not args.dataset.strip():
        raise ValueError("--dataset is required for non-conflict conditions")

    backend = _infer_backend_from_model(args.judge)
    os.environ["CONTEXT_LLM_BACKEND"] = backend
    print(f"[analyze_data] inferred backend={backend} for judge={args.judge}")

    print(f"[analyze_data] loading dataset...")
    df = _load_dataset_records(args.dataset, args.max_rows)
    print(f"[analyze_data] loaded {len(df)} records")

    if args.condition == "none":
        summary = _condition_none(df)
        summary["dataset"] = args.dataset
        _print_summary(summary)
        with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"[analyze_data] Saved summary: {output_dir / 'summary.json'}")
        return

    use_web_search = condition_requires_web_search(args.condition)
    print(f"[analyze_data] condition_requires_web_search({args.condition})={use_web_search}")

    if use_web_search and backend != "openai":
        raise RuntimeError("This condition requires OpenAI Responses API web_search support")

    if backend == "openai":
        query_fn = _query_openai_responses
    else:
        query_fn = _query_no_web

    # Resolve per-row claim dates for stale mode
    claim_dates: list[str | None] | None = None
    if args.condition == "stale":
        claim_dates = _resolve_claim_dates(df, args.dataset)

    print(f"[analyze_data] starting LLM judging...")
    summary, rows = _run_llm_condition(
        df=df,
        condition=args.condition,
        model=args.judge,
        query_fn=query_fn,
        use_web_search=use_web_search,
        workers=args.workers,
        claim_dates=claim_dates,
    )
    summary["dataset"] = args.dataset

    _print_summary(summary)

    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "per_claim_judgments.jsonl"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(rows_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"[analyze_data] Saved summary: {summary_path}")
    print(f"[analyze_data] Saved judgments: {rows_path}")


if __name__ == "__main__":
    import sys
    try:
        main()
    except Exception as e:
        print(f"[analyze_data] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
