#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
from datetime import datetime
from glob import glob
from hashlib import md5
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
LOCAL_CORE_DIR = SCRIPTS_DIR / "context_core"
DEFAULT_STRUCTURAL_SHARDS_DIR = "/network/scratch/k/kondrupe/credibench-neighbors_serving_shards"
DATA_STATS_DIR = ROOT_DIR / "data_stats"
EVAL_REPORTS_DIR = ROOT_DIR / "eval"


def _python_path_setup() -> None:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RARR reproduction, on local CSV/JSONL or Hugging Face datasets."
    )
    parser.add_argument("dataset_path", nargs="?", help="Path to local dataset (.csv or .jsonl)")
    parser.add_argument("run_id", nargs="?", help="Run ID (default: timestamp-based)")
    parser.add_argument("max_rows", nargs="?", type=int, default=50, help="Max rows to evaluate")

    parser.add_argument("--structural", action="store_true", help="Enable structural graph context")
    parser.add_argument(
        "--structural-shards-dir",
        default=os.getenv("RARR_STRUCTURAL_SHARDS_DIR", DEFAULT_STRUCTURAL_SHARDS_DIR),
        help="Directory containing hook serving shards",
    )
    parser.add_argument("--show-prompts", action="store_true", help="Print active RARR prompts and exit")

    parser.add_argument("--hf-dataset", default="", help="HF dataset repo, e.g. ComplexDataLab/Misinfo_Datasets")
    parser.add_argument("--hf-config", default="default", help="HF dataset config")
    parser.add_argument("--hf-split", default="train", help="HF split")
    parser.add_argument("--hf-claim-col", default="claim", help="Claim column for HF mode")
    parser.add_argument("--hf-label-col", default="label", help="Ground-truth label column for HF mode")
    parser.add_argument("--factcheck-model", default="", help="Override the fact-check claim processor model")
    parser.add_argument("--rarr-model", default="", help="Override the RARR retriever/verifier model")

    return parser.parse_args()


def _show_prompts() -> None:
    _python_path_setup()
    from context_core.prompts import functional_prompt

    print("=== QGEN_PROMPT ===")
    print(functional_prompt.QGEN_PROMPT)
    print("\n=== AGREEMENT_GATE_PROMPT ===")
    print(functional_prompt.AGREEMENT_GATE_PROMPT)


def _validate_runtime(args: argparse.Namespace) -> None:
    if not LOCAL_CORE_DIR.exists():
        raise FileNotFoundError(f"Local vendored core directory not found at {LOCAL_CORE_DIR}")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    provider = os.getenv("RARR_SEARCH_PROVIDER", "serper").strip().lower()
    if provider == "serper" and not os.getenv("SERPER_API_KEY"):
        raise RuntimeError("RARR_SEARCH_PROVIDER=serper requires SERPER_API_KEY")
    if provider not in {"auto", "serper", "duckduckgo"}:
        raise RuntimeError(
            f"Unsupported RARR_SEARCH_PROVIDER={provider}. Use one of: auto, serper, duckduckgo"
        )

    if args.structural:
        shards_dir = Path(args.structural_shards_dir)
        if not shards_dir.is_dir():
            raise RuntimeError(
                f"Structural mode requested, but serving shards directory not found: {shards_dir}"
            )
        if not (shards_dir / "_meta.json").exists():
            print(
                f"Warning: {(shards_dir / '_meta.json')} is missing; shard count defaults may be used."
            )
        os.environ["RARR_STRUCTURAL_MODE"] = "1"
        os.environ["RARR_STRUCTURAL_SHARDS_DIR"] = str(shards_dir)
        os.environ.setdefault("RARR_STRUCTURAL_HOOK_PATH", str((ROOT_DIR / "scripts" / "hook.py").resolve()))
        print(f"Structural mode enabled (shards: {shards_dir})")


def _load_local_dataset(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    raise ValueError("Only .csv and .jsonl are supported")


def _load_hf_dataset(repo: str, config: str, split: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for --hf-dataset mode. "
            "Install it in ctxt-env, e.g. uv pip install datasets"
        ) from exc

    ds = load_dataset(repo, config, split=split)
    return ds.to_pandas()


def _load_hf_dataset_dict(repo: str, config: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for --hf-dataset mode. "
            "Install it in ctxt-env, e.g. uv pip install datasets"
        ) from exc

    return load_dataset(repo, config)


def _normalize_label(value: object) -> str:
    if value is None:
        return "unverified"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if int(value) == 1:
            return "true"
        if int(value) == 0:
            return "false"
        return "unverified"
    low = str(value).strip().lower()
    if low in {"true", "supported", "support", "factual", "real", "correct"}:
        return "true"
    if low in {"false", "refuted", "fake", "incorrect"}:
        return "false"
    return "unverified"


def _infer_pred_label(payload: dict) -> str:
    claims = payload.get("claims", {}) or {}
    num_true = int(claims.get("numTrueClaims", 0) or 0)
    num_false = int(claims.get("numFalseClaims", 0) or 0)
    num_mixed = int(claims.get("numMixedClaims", 0) or 0)
    num_undefined = int(claims.get("numUndefinedClaims", 0) or 0)

    if num_mixed > 0 or num_undefined > 0:
        return "unverified"
    if num_true > 0 and num_false == 0:
        return "true"
    if num_false > 0 and num_true == 0:
        return "false"
    return "true" if bool(payload.get("result", False)) else "false"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "unknown"


def _read_latest_state(truth_output_dir: Path, sample_name: str) -> dict:
    result_path = truth_output_dir / f"{sample_name}.jsonl"
    if not result_path.exists():
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


def _sample_name(dataset_name: str, index: int) -> str:
    return f"dataset_na_{index}"


def _normalize_decision_label(label: object) -> str:
    if not isinstance(label, str):
        return "ambiguous"

    low = label.strip().lower()
    if low in {"agrees", "true", "supports", "support"}:
        return "true"
    if low in {"disagrees", "false", "refutes", "refute", "contradicts", "contradict"}:
        return "false"
    if low == "unverifiable":
        return "unverifiable"
    if low in {"ambiguous", "irrelevant", "unknown"}:
        return "ambiguous"
    return "ambiguous"


def _infer_prediction_bucket_from_state(state: dict) -> str:
    details = state.get("detail") or state.get("log") or []
    if not isinstance(details, list) or not details:
        return "unverifiable"

    normalized: list[str] = []
    for item in details:
        labels = item.get("labels")
        if not isinstance(labels, list) or not labels:
            factuality = item.get("factuality")
            if factuality is True:
                normalized.append("true")
            elif factuality is False:
                normalized.append("false")
            else:
                normalized.append("unverifiable")
            continue
        normalized.extend(_normalize_decision_label(label) for label in labels)

    uniq = set(normalized)
    if uniq == {"true"}:
        return "true"
    if uniq == {"false"}:
        return "false"
    if uniq == {"unverifiable"}:
        return "unverifiable"
    if uniq == {"ambiguous"}:
        return "ambiguous"
    return "ambiguous"


def _collect_prediction_buckets(output_dir: Path, total_rows: int, dataset_name: str) -> dict[int, str]:
    truth_dir = output_dir / "truth"
    pred_by_idx: dict[int, str] = {}
    for idx in range(total_rows):
        state = _read_latest_state(truth_dir, _sample_name(dataset_name, idx))
        if state:
            pred_by_idx[idx] = _infer_prediction_bucket_from_state(state)
    return pred_by_idx


def _write_dataset_split_stats(
    hf_dataset: str,
    hf_config: str,
    label_col: str,
) -> tuple[Path, dict]:
    DATA_STATS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dict = _load_hf_dataset_dict(hf_dataset, hf_config)
    classes = ["true", "false", "unverified"]

    splits_report: dict[str, dict[str, int]] = {}
    for split_name, split_ds in dataset_dict.items():
        df = split_ds.to_pandas()
        labels = [_normalize_label(value) for value in df[label_col].tolist()] if label_col in df.columns else []
        splits_report[split_name] = {
            "total": len(df),
            "true": sum(1 for value in labels if value == "true"),
            "false": sum(1 for value in labels if value == "false"),
            "unverified": sum(1 for value in labels if value == "unverified"),
        }
        for label in classes:
            splits_report[split_name].setdefault(label, 0)

    report = {
        "dataset": hf_dataset,
        "config": hf_config,
        "label_column": label_col,
        "splits": splits_report,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    report_path = DATA_STATS_DIR / f"{_slugify(hf_dataset)}.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report_path, report


def _load_model_config(config_path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to read model config metadata") from exc

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _model_report_identity(config_path: Path) -> tuple[str, dict]:
    config = _load_model_config(config_path)
    global_config = config.get("global_config", {}) if isinstance(config, dict) else {}
    factcheck_model = str(global_config.get("factcheck_gpt_model", "unknown"))
    rarr_model = str(global_config.get("rarr_model", "unknown"))
    model_id = _slugify(f"factcheck-{factcheck_model}__rarr-{rarr_model}")
    return model_id, {
        "factcheck_gpt_model": factcheck_model,
        "rarr_model": rarr_model,
        "config_path": str(config_path),
    }


def _write_model_eval_report(
    run_id: str,
    dataset_name: str,
    hf_split: str,
    labels: list[str],
    prediction_buckets: dict[int, str],
    config_path: Path,
) -> tuple[Path, dict]:
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_id, model_info = _model_report_identity(config_path)
    report_path = EVAL_REPORTS_DIR / f"{model_id}.json"

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    else:
        report = {
            "model": model_info,
            "runs": {},
        }

    per_category = {
        "true": {
            "total": 0,
            "predicted_correctly": 0,
            "predicted_other_binary": 0,
            "predicted_ambiguous": 0,
            "predicted_unverifiable": 0,
        },
        "false": {
            "total": 0,
            "predicted_correctly": 0,
            "predicted_other_binary": 0,
            "predicted_ambiguous": 0,
            "predicted_unverifiable": 0,
        },
        "unverified": {
            "total": 0,
            "predicted_correctly": 0,
            "predicted_other_binary": 0,
            "predicted_ambiguous": 0,
            "predicted_unverifiable": 0,
        },
    }

    for idx, gold in enumerate(labels):
        pred = prediction_buckets.get(idx)
        if gold not in per_category or pred is None:
            continue
        per_category[gold]["total"] += 1
        if gold == "true":
            if pred == "true":
                per_category[gold]["predicted_correctly"] += 1
            elif pred == "false":
                per_category[gold]["predicted_other_binary"] += 1
            elif pred == "ambiguous":
                per_category[gold]["predicted_ambiguous"] += 1
            elif pred == "unverifiable":
                per_category[gold]["predicted_unverifiable"] += 1
        elif gold == "false":
            if pred == "false":
                per_category[gold]["predicted_correctly"] += 1
            elif pred == "true":
                per_category[gold]["predicted_other_binary"] += 1
            elif pred == "ambiguous":
                per_category[gold]["predicted_ambiguous"] += 1
            elif pred == "unverifiable":
                per_category[gold]["predicted_unverifiable"] += 1
        else:
            if pred == "unverifiable":
                per_category[gold]["predicted_correctly"] += 1
            elif pred in {"true", "false"}:
                per_category[gold]["predicted_other_binary"] += 1
            elif pred == "ambiguous":
                per_category[gold]["predicted_ambiguous"] += 1

    report["runs"][run_id] = {
        "dataset": dataset_name,
        "split": hf_split,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows_scored": sum(1 for idx in range(len(labels)) if idx in prediction_buckets),
        "per_category": per_category,
    }

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report_path, report


def _collect_predictions(output_dir: Path) -> dict[int, str]:
    pred_by_idx: dict[int, str] = {}
    for path in glob(str(output_dir / "*" / "eval_result.json")):
        dirname = os.path.basename(os.path.dirname(path))
        match = re.search(r"_(\d+)_[0-9a-f]{32}$", dirname)
        if not match:
            continue
        idx = int(match.group(1))
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        pred_by_idx[idx] = _infer_pred_label(payload)
    return pred_by_idx


def _write_ground_truth_report(output_dir: Path, labels: list[str], preds: dict[int, str]) -> tuple[Path, dict]:
    classes = ["true", "false", "unverified"]
    comparable = [(gold, preds[idx]) for idx, gold in enumerate(labels) if idx in preds]

    correct = sum(1 for g, p in comparable if g == p)
    incorrect = len(comparable) - correct
    accuracy = correct / len(comparable) if comparable else None

    confusion = {gold: {pred: 0 for pred in classes} for gold in classes}
    for gold, pred in comparable:
        g = gold if gold in classes else "unverified"
        p = pred if pred in classes else "unverified"
        confusion[g][p] += 1

    report = {
        "total_rows": len(labels),
        "rows_with_predictions": len(preds),
        "rows_scored": len(comparable),
        "correct": correct,
        "incorrect": incorrect,
        "ground_truth_counts": {
            "true": sum(1 for x in labels if x == "true"),
            "false": sum(1 for x in labels if x == "false"),
            "unverified": sum(1 for x in labels if x == "unverified"),
        },
        "prediction_counts": {
            "true": sum(1 for x in preds.values() if x == "true"),
            "false": sum(1 for x in preds.values() if x == "false"),
            "unverified": sum(1 for x in preds.values() if x == "unverified"),
        },
        "accuracy": accuracy,
        "confusion": confusion,
    }

    report_path = output_dir / "ground_truth_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report_path, report


def _default_run_id(hf_dataset: str) -> str:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ("rarr-hf-" if hf_dataset else "rarr-dataset-") + now


def run(args: argparse.Namespace) -> None:
    _validate_runtime(args)

    if not args.hf_dataset and not args.dataset_path:
        raise ValueError("dataset_path is required unless --hf-dataset is provided")

    run_id = args.run_id or _default_run_id(args.hf_dataset)
    max_rows = args.max_rows
    output_dir = ROOT_DIR / "eval_results" / "custom" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if os.getenv("RARR_STRUCTURAL_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        os.environ.setdefault(
            "RARR_STRUCTURAL_CACHE_FILE",
            str((output_dir / "structural_domain_cache.json").resolve()),
        )

    if args.hf_dataset:
        df = _load_hf_dataset(args.hf_dataset, args.hf_config, args.hf_split)
    else:
        dataset_path = str(Path(args.dataset_path).resolve())
        if not Path(dataset_path).is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        df = _load_local_dataset(dataset_path)

    if "prompt" not in df.columns:
        if args.hf_dataset and args.hf_claim_col in df.columns:
            df["prompt"] = df[args.hf_claim_col]
        elif "claim" in df.columns:
            df["prompt"] = df["claim"]
        else:
            raise ValueError("Dataset must include 'prompt' (or 'claim').")

    if "response" not in df.columns:
        if args.hf_dataset:
            df["response"] = df["prompt"]
        else:
            raise ValueError("Dataset must include 'response'.")

    if "source" not in df.columns:
        df["source"] = "hf-dataset" if args.hf_dataset else "custom-dataset"

    records = df[["source", "prompt", "response"]].dropna().head(max_rows).reset_index(drop=True)
    if records.empty:
        raise ValueError("No valid rows found after filtering null prompt/response")

    labels = None
    if args.hf_dataset and args.hf_label_col in df.columns:
        labels = [_normalize_label(v) for v in df[args.hf_label_col].head(max_rows).tolist()]

    dataset_name = str(df["source"].iloc[0]) if not df.empty else (args.hf_dataset or "custom-dataset")

    _python_path_setup()
    from context_core.benchmark import evaluate_free_text_with_auto_checker

    solver_args = argparse.Namespace(
        user_src=str((LOCAL_CORE_DIR / "solvers").resolve()),
        config=str((LOCAL_CORE_DIR / "config" / "rarr_web_service_config.yaml").resolve()),
        output=str((output_dir / "truth").resolve()),
        openai_apikey=os.getenv("OPENAI_API_KEY"),
        factcheck_model=args.factcheck_model,
        rarr_model=args.rarr_model,
    )

    evaluate_free_text_with_auto_checker(
        records.to_dict("records"),
        response_column_name="dataset_response",
        args=solver_args,
        projectdir=str(output_dir.resolve()),
    )

    print(f"Run complete: {run_id}")
    print(f"Rows processed: {len(records)}")
    print(f"Output: {output_dir}")

    if args.hf_dataset:
        stats_path, _ = _write_dataset_split_stats(args.hf_dataset, args.hf_config, args.hf_label_col)
        print(f"Dataset split stats: {stats_path}")

    if labels is not None:
        preds = _collect_predictions(output_dir)
        prediction_buckets = _collect_prediction_buckets(output_dir, len(labels), dataset_name)
        report_path, report = _write_ground_truth_report(output_dir, labels, preds)
        eval_report_path, _ = _write_model_eval_report(
            run_id=run_id,
            dataset_name=args.hf_dataset or dataset_name,
            hf_split=args.hf_split,
            labels=labels,
            prediction_buckets=prediction_buckets,
            config_path=Path(solver_args.config),
        )
        print("Ground-truth report:")
        print(f"  path: {report_path}")
        print(f"  rows scored: {report['rows_scored']}")
        print(f"  correct: {report['correct']}")
        print(f"  incorrect: {report['incorrect']}")
        print(
            "  gt counts: "
            f"true={report['ground_truth_counts']['true']}, "
            f"false={report['ground_truth_counts']['false']}, "
            f"unverified={report['ground_truth_counts']['unverified']}"
        )
        print(f"  accuracy: {report['accuracy']}")
        print(f"Model eval report: {eval_report_path}")
    elif args.hf_dataset:
        print(f"No '{args.hf_label_col}' column found in HF split; skipping ground-truth report.")


def main() -> None:
    args = parse_args()
    if args.show_prompts:
        _show_prompts()
        return
    run(args)


if __name__ == "__main__":
    main()
