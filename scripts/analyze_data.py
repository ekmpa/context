from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

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
    """Load environment variables from .env file if it exists."""
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        value = value.strip()
                        # Strip surrounding quotes (both single and double)
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        os.environ.setdefault(key.strip(), value)


# Load .env at module import time
_load_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze claim datasets with condition-specific counting and LLM-as-judge modes.")
    parser.add_argument("--dataset", default="", help="Dataset name under data/<dataset> or a direct file path (.jsonl/.csv)")
    parser.add_argument(
        "--condition",
        required=True,
        choices=sorted(SUPPORTED_CONDITIONS),
        help="Analysis condition: none | conflict | stale | opinion | unverif | ambig",
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
    lower = (model_name or "").strip().lower()
    if lower.startswith(("qwen", "llama")):
        return "hf-local"
    return "openai"


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
    seed: int | None,
) -> Path:
    if explicit_path.strip():
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Run log not found: {path}")
        return path

    dataset_filter = dataset.strip().lower()
    model_filter = rarr_model.strip().lower()
    # Conflict analysis is intended for fact-checking evidence behavior from the
    # standard raw pipeline unless explicitly overridden.
    condition_filter = (rarr_condition.strip().lower() or "raw")

    candidates: list[tuple[int, int, float, Path]] = []
    for meta_path in (ROOT_DIR / "eval_results" / "custom").glob("rarr-*/meta_eval.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta_dataset = str(meta.get("dataset", "") or "").strip().lower()
        meta_model = str(((meta.get("model") or {}).get("rarr_model", "") or "")).strip().lower()
        meta_condition = str(meta.get("condition", "") or "").strip().lower()
        meta_seed = meta.get("seed", None)
        rows_total = int(meta.get("rows_total", 0) or 0)

        if dataset_filter and meta_dataset and meta_dataset != dataset_filter:
            continue
        model_match = 1
        if model_filter:
            model_match = 1 if (meta_model and meta_model == model_filter) else 0
        if condition_filter and meta_condition and meta_condition != condition_filter:
            continue
        if condition_filter and not meta_condition:
            # Older runs may not store condition in meta_eval. Keep them as a
            # fallback candidate; ranking below prefers explicit condition
            # matches and larger runs.
            pass
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
        # Rank by rows_total first (prefer fuller runs like 600 over 10/50
        # smoke tests), then prefer model matches, then recency.
        candidates.append((rows_total, model_match, meta_path.stat().st_mtime, model_logs[0].resolve()))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    resolved = [path for _, _, _, path in candidates]
    if not candidates:
        raise FileNotFoundError(
            "No matching run log found under eval_results/custom for the requested filters. "
            "Provide --run-log <path/to/factcheck-*.json> or relax dataset/model/condition/seed filters."
        )
    return resolved[0]


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
    print(
        f"[analyze_data] Running condition='{condition}' on {total} rows with judge='{model}' "
        f"(workers={resolved_workers})"
    )

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

    claims = df["claim"].tolist()
    processed = 0
    if resolved_workers == 1:
        for idx, claim in enumerate(claims):
            _, row = _process_one(idx, claim)
            label_counts[row["judge_label"]] += 1
            rows[idx] = row
            processed += 1
            if processed % 25 == 0 or processed == total:
                print(f"[analyze_data]   processed {processed}/{total}")
    else:
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
