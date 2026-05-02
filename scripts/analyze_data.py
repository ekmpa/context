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
    parser.add_argument("--dataset", required=True, help="Dataset name under data/<dataset> or a direct file path (.jsonl/.csv)")
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
    parser.add_argument("--third-party-ratings-file", default="", help="Accepted for IO compatibility; not used yet")
    return parser.parse_args()


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


def _run_llm_condition(
    *,
    df: pd.DataFrame,
    condition: str,
    model: str,
    query_fn: Callable[[str, str, str, bool], str],
    use_web_search: bool,
    workers: int,
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
        system_prompt, user_prompt = judge_prompts(condition, claim)
        raw = query_fn(model, system_prompt, user_prompt, use_web_search)
        norm = normalize_judge_label(condition, raw)
        row = {
            "index": idx,
            "claim": claim,
            "judge_label": norm,
            "judge_raw": raw,
        }
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

    if summary.get("placeholder"):
        print(summary.get("message", "placeholder"))


def main() -> None:
    args = parse_args()
    print(f"[analyze_data] dataset={args.dataset}, condition={args.condition}, max_rows={args.max_rows}, workers={args.workers}")

    backend = _infer_backend_from_model(args.judge)
    os.environ["CONTEXT_LLM_BACKEND"] = backend
    print(f"[analyze_data] inferred backend={backend} for judge={args.judge}")

    print(f"[analyze_data] loading dataset...")
    df = _load_dataset_records(args.dataset, args.max_rows)
    print(f"[analyze_data] loaded {len(df)} records")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analyze_data] output_dir={output_dir}")

    if args.condition == "none":
        summary = _condition_none(df)
        summary["dataset"] = args.dataset
        _print_summary(summary)
        with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"[analyze_data] Saved summary: {output_dir / 'summary.json'}")
        return

    if args.condition == "conflict":
        summary = _condition_conflict_placeholder(df)
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

    print(f"[analyze_data] starting LLM judging...")
    summary, rows = _run_llm_condition(
        df=df,
        condition=args.condition,
        model=args.judge,
        query_fn=query_fn,
        use_web_search=use_web_search,
        workers=args.workers,
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
