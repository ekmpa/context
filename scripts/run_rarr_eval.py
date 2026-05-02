import argparse
from collections import Counter
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
LOCAL_CORE_DIR = SCRIPTS_DIR / "context_core"
DEFAULT_STRUCTURAL_SHARDS_DIR = "/network/scratch/k/kondrupe/credibench-neighbors_serving_shards"
DATA_STATS_DIR = ROOT_DIR / "data_stats"
MODEL_BACKEND_PREFIX_MAP: dict[str, tuple[str, ...]] = {
    # Extend here as new model families are added.
    "hf-local": ("qwen", "llama"),
    "openai": ("gpt", "o1", "o3", "o4", "text-"),
}
CLAIM_COUNTER_KEYS = (
    "numFalseClaims",
    "numMixedClaims",
    "numTrueClaims",
    "numUndefinedClaims",
)

TRUE_LABEL_ALIASES = {
    "true",
    "supported",
    "support",
    "supports",
    "entailment",
    "entails",
    "factual",
    "real",
    "correct",
    "accurate",
    "verified",
    "yes",
    "mostly-true",
    "mostly true",
    "half-true",
    "half true",
}

FALSE_LABEL_ALIASES = {
    "false",
    "refuted",
    "refute",
    "refutes",
    "contradiction",
    "contradict",
    "contradicts",
    "fake",
    "incorrect",
    "inaccurate",
    "pants-fire",
    "pants fire",
    "pants-on-fire",
    "barely-true",
    "barely true",
    "no",
}

UNVERIFIED_LABEL_ALIASES = {
    "unverified",
    "unverifiable",
    "undefined",
    "unknown",
    "uncertain",
    "not enough info",
    "not enough information",
    "nei",
    "neutral",
    "ambiguous",
    "irrelevant",
    "mixed",
    "mixture",
    "partly true",
    "partly false",
    "misleading",
}

COMPACT_LABEL_MAP = {
    "notenoughinfo": "unverified",
    "pantsonfire": "false",
    "mostlytrue": "true",
    "halftrue": "true",
    "barelytrue": "false",
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _condition_hyperparameter_defaults(condition: str) -> tuple[int, int]:
    if condition in {"structural", "third-party", "raw"}:
        return 5, 5
    return 5, 5


def _resolve_runtime_hyperparameters(
    *,
    condition: str,
    num_rounds_qgen: int | None,
    max_evidences_per_question: int | None,
) -> tuple[int, int]:
    default_rounds, default_evidences = _condition_hyperparameter_defaults(condition)
    resolved_rounds = num_rounds_qgen
    if resolved_rounds is None:
        resolved_rounds = _env_int("RARR_NUM_ROUNDS_QGEN", default_rounds)
    resolved_evidences = max_evidences_per_question
    if resolved_evidences is None:
        resolved_evidences = _env_int("RARR_MAX_EVIDENCES_PER_QUESTION", default_evidences)
    return max(1, int(resolved_rounds)), max(1, int(resolved_evidences))


def _parse_optional_max_rows(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None

    text = str(value).strip().lower()
    if text in {"", "none", "all", "null"}:
        return None

    parsed = int(text)
    if parsed < 1:
        raise ValueError("--max-rows must be a positive integer or one of: none, all")
    return parsed


def _python_path_setup() -> None:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RARR reproduction, on local CSV/JSONL or Hugging Face datasets."
    )
    parser.add_argument("dataset_path", nargs="?", help="Path to local dataset (.csv or .jsonl)")
    parser.add_argument("run_id", nargs="?", help="Run ID (default: timestamp-based)")
    parser.add_argument("max_rows", nargs="?", type=int, default=None, help="Max rows to evaluate")
    parser.add_argument(
        "--max-rows",
        dest="max_rows_flag",
        type=str,
        default=None,
        help="Max rows to evaluate (named flag, preferred for scripting)",
    )

    parser.add_argument(
        "--condition",
        choices=["raw", "structural", "third-party", "source_attr"],
        default=os.getenv("RARR_CONDITION", "raw"),
        help="Run mode: raw (web only), structural (web + graph context), or third-party (web + domain ratings)",
    )
    parser.add_argument("--structural", action="store_true", help="Enable structural graph context")
    parser.add_argument(
        "--structural-shards-dir",
        default=os.getenv("RARR_STRUCTURAL_SHARDS_DIR", DEFAULT_STRUCTURAL_SHARDS_DIR),
        help="Directory with structural serving shards",
    )
    parser.add_argument(
        "--third-party-ratings-file",
        default=os.getenv("RARR_THIRD_PARTY_RATINGS_FILE", str((ROOT_DIR / "data" / "domain_ratings.csv").resolve())),
        help="CSV file with domain trust scores for third-party mode",
    )
    parser.add_argument("--show-prompts", action="store_true", help="Show the active RARR prompts and exit")

    parser.add_argument("--hf-dataset", default="", help="Hugging Face dataset, for example ComplexDataLab/Misinfo_Datasets")
    parser.add_argument("--hf-config", default="default", help="Hugging Face dataset config")
    parser.add_argument("--hf-split", default="train", help="Hugging Face split")
    parser.add_argument("--hf-claim-col", default="claim", help="Claim column for Hugging Face runs")
    parser.add_argument("--hf-label-col", default="label", help="Label column for Hugging Face runs")
    parser.add_argument(
        "--hf-origin-col",
        default="dataset",
        help="Column that stores the original source dataset name",
    )
    parser.add_argument(
        "--hf-origin-value",
        default="",
        help="Keep only rows with one origin value from --hf-origin-col",
    )
    parser.add_argument(
        "--hf-list-origin-values",
        action="store_true",
        help="List row counts by origin value and exit",
    )
    parser.add_argument("--factcheck-model", default="", help="Override the claim-processing model")
    parser.add_argument("--rarr-model", default="", help="Override the retriever/verifier model")
    parser.add_argument(
        "--num-rounds-qgen",
        type=int,
        default=None,
        help="How many question-generation rounds to run. Default: env override, else 5.",
    )
    parser.add_argument(
        "--max-evidences-per-question",
        type=int,
        default=None,
        help="How many evidence snippets to pass to the verifier per claim. Default: env override, else 5.",
    )

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
        raise FileNotFoundError(f"Couldn't find the local core package at {LOCAL_CORE_DIR}")

    backend = os.getenv("CONTEXT_LLM_BACKEND", "openai").strip().lower()
    if backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI backend")
    elif backend not in {"hf-local", "hf_local", "local"}:
        raise RuntimeError(
            f"Unsupported CONTEXT_LLM_BACKEND={backend}. Use 'openai' or 'hf-local'."
        )

    provider = os.getenv("RARR_SEARCH_PROVIDER", "serper").strip().lower()
    if provider == "serper" and not os.getenv("SERPER_API_KEY"):
        raise RuntimeError("Set SERPER_API_KEY to use the Serper search provider")
    if provider not in {"auto", "serper", "duckduckgo"}:
        raise RuntimeError(
            f"Unsupported RARR_SEARCH_PROVIDER={provider}. Use 'auto', 'serper', or 'duckduckgo'."
        )

    condition = _resolve_condition(args)

    if condition == "structural":
        shards_dir = Path(args.structural_shards_dir)
        if not shards_dir.is_dir():
            raise RuntimeError(
                f"Structural mode needs a serving shards directory, but none was found at {shards_dir}"
            )
        if not (shards_dir / "_meta.json").exists():
            print(
                f"Warning: {(shards_dir / '_meta.json')} is missing, so shard defaults may be used."
            )
        os.environ["RARR_STRUCTURAL_MODE"] = "1"
        os.environ["RARR_STRUCTURAL_SHARDS_DIR"] = str(shards_dir)
        os.environ.setdefault("RARR_STRUCTURAL_HOOK_PATH", str((ROOT_DIR / "scripts" / "hook.py").resolve()))
        print(f"Structural mode is on (shards: {shards_dir})")

    if condition == "third-party":
        ratings_path = Path(args.third_party_ratings_file)
        if not ratings_path.is_file():
            raise RuntimeError(
                f"Third-party mode needs a ratings file, but none was found at {ratings_path}"
            )


def _infer_backend_from_model(model_name: str) -> str:
    lower_name = (model_name or "").strip().lower()
    for backend, prefixes in MODEL_BACKEND_PREFIX_MAP.items():
        if any(lower_name.startswith(prefix) for prefix in prefixes):
            return backend
    return "openai"


def _configure_backend_environment(args: argparse.Namespace) -> str:
    # rarr_model can come from CLI, env, or fall back to default config behavior.
    chosen_model = (
        (args.rarr_model or "").strip()
        or os.getenv("RARR_MODEL", "").strip()
        or "gpt-3.5-turbo-instruct"
    )
    inferred_backend = _infer_backend_from_model(chosen_model)
    os.environ["CONTEXT_LLM_BACKEND"] = inferred_backend
    return inferred_backend


def _resolve_condition(args: argparse.Namespace) -> str:
    # Backward compatibility: --structural forces structural mode.
    if getattr(args, "structural", False):
        return "structural"
    condition = getattr(args, "condition", "raw")
    return condition


def _configure_condition_environment(args: argparse.Namespace) -> str:
    condition = _resolve_condition(args)
    os.environ["RARR_CONDITION"] = condition
    os.environ["RARR_STRUCTURAL_MODE"] = "1" if condition == "structural" else "0"
    os.environ["RARR_THIRD_PARTY_MODE"] = "1" if condition == "third-party" else "0"
    os.environ["RARR_SOURCE_ATTR_MODE"] = "1" if condition == "source_attr" else "0"
    if condition == "third-party":
        os.environ["RARR_THIRD_PARTY_RATINGS_FILE"] = str(Path(args.third_party_ratings_file).resolve())
    return condition


def _load_local_dataset(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    raise ValueError("Use a .csv or .jsonl dataset file")


def _hf_import():
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it in ctxt-env with: uv pip install datasets"
        ) from exc


def _load_hf_dataset(repo: str, config: str, split: str) -> pd.DataFrame:
    return _hf_import()(repo, config, split=split).to_pandas()


def _load_hf_dataset_dict(repo: str, config: str):
    return _hf_import()(repo, config)


def _normalize_label_with_match(value: object) -> tuple[str, bool, str]:
    if value is None:
        return "unverified", True, "none"
    if isinstance(value, bool):
        return ("true" if value else "false"), True, "bool"
    if isinstance(value, (int, float)):
        if int(value) == 1:
            return "true", True, "numeric"
        if int(value) == 0:
            return "false", True, "numeric"
        return "unverified", False, "numeric"

    low = str(value).strip().lower()
    if low in TRUE_LABEL_ALIASES:
        return "true", True, "direct"
    if low in FALSE_LABEL_ALIASES:
        return "false", True, "direct"
    if low in UNVERIFIED_LABEL_ALIASES:
        return "unverified", True, "direct"

    token = re.sub(r"[^a-z]+", " ", low).strip()
    if token in TRUE_LABEL_ALIASES:
        return "true", True, "tokenized"
    if token in FALSE_LABEL_ALIASES:
        return "false", True, "tokenized"
    if token in UNVERIFIED_LABEL_ALIASES:
        return "unverified", True, "tokenized"

    compact = token.replace(" ", "")
    if compact in COMPACT_LABEL_MAP:
        return COMPACT_LABEL_MAP[compact], True, "compact"
    return "unverified", False, "fallback"


def _normalize_label(value: object) -> str:
    normalized, _, _ = _normalize_label_with_match(value)
    return normalized


def _build_label_audit(raw_labels: list[object], normalized_labels: list[str]) -> dict:
    raw_counter: Counter[str] = Counter()
    unknown_counter: Counter[str] = Counter()
    normalized_counter: Counter[str] = Counter(normalized_labels)
    matched_count = 0

    for value in raw_labels:
        raw_text = str(value).strip()
        raw_display = raw_text if raw_text else "<empty>"
        raw_counter[raw_display] += 1
        _, matched, _ = _normalize_label_with_match(value)
        if matched:
            matched_count += 1
        else:
            unknown_counter[raw_display] += 1

    rows_considered = len(raw_labels)
    return {
        "rows_considered": rows_considered,
        "raw_unique": len(raw_counter),
        "matched_count": matched_count,
        "fallback_count": rows_considered - matched_count,
        "normalized_counts": {
            "true": int(normalized_counter.get("true", 0)),
            "false": int(normalized_counter.get("false", 0)),
            "unverified": int(normalized_counter.get("unverified", 0)),
        },
        "top_raw_labels": [
            {"label": label, "count": count}
            for label, count in raw_counter.most_common(10)
        ],
        "top_fallback_labels": [
            {"label": label, "count": count}
            for label, count in unknown_counter.most_common(10)
        ],
    }


def _normalize_text_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "na", "none", "null", "nan"}:
        return None
    return text


def _resolve_label_column(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    candidates = [
        preferred,
        "label",
        "labels",
        "veracity",
        "verdict",
        "gold_label",
        "annotation",
        "stance",
    ]
    for col in candidates:
        if col and col in df.columns:
            return col
    return None


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


def _normalize_decision_label(label: object) -> str:
    if not isinstance(label, str):
        return "ambiguous"

    low = label.strip().lower()
    if low in {"agrees", "true", "supports", "support"}:
        return "true"
    if low in {"disagrees", "false", "refutes", "refute", "contradicts", "contradict"}:
        return "false"
    if low in {"unverifiable", "unverified", "undefined"}:
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


def _annotate_eval_entries(entries: list[dict]) -> tuple[dict[int, str], dict[int, str]]:
    pred_labels: dict[int, str] = {}
    pred_buckets: dict[int, str] = {}
    for idx, entry in enumerate(entries):
        payload = {
            "claims": entry.get("claims", {}) or {},
            "result": entry.get("result"),
        }
        pred_label = _infer_pred_label(payload)
        pred_bucket = _infer_prediction_bucket_from_state({"detail": entry.get("detail", [])})
        entry["pred_label"] = pred_label
        entry["pred_bucket"] = pred_bucket
        pred_labels[idx] = pred_label
        pred_buckets[idx] = pred_bucket
    return pred_labels, pred_buckets


def _aggregate_claim_counters(entries: list[dict]) -> dict[str, int]:
    totals = {key: 0 for key in CLAIM_COUNTER_KEYS}
    for entry in entries:
        claims = entry.get("claims") if isinstance(entry, dict) else None
        if not isinstance(claims, dict):
            continue
        for key in CLAIM_COUNTER_KEYS:
            totals[key] += int(claims.get(key, 0) or 0)
    return totals


def _write_dataset_split_stats(
    hf_dataset: str,
    hf_config: str,
    label_col: str,
    origin_col: str,
) -> tuple[Path, dict]:
    DATA_STATS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dict = _load_hf_dataset_dict(hf_dataset, hf_config)
    classes = ["true", "false", "unverified"]

    def _label_counts(values: list[object]) -> dict[str, int]:
        labels = [_normalize_label(value) for value in values]
        counts = {
            "total": len(values),
            "true": sum(1 for value in labels if value == "true"),
            "false": sum(1 for value in labels if value == "false"),
            "unverified": sum(1 for value in labels if value == "unverified"),
        }
        for label in classes:
            counts.setdefault(label, 0)
        return counts

    splits_report: dict[str, dict[str, int]] = {}
    per_origin: dict[str, dict[str, object]] = {}
    for split_name, split_ds in dataset_dict.items():
        df = split_ds.to_pandas()
        if label_col in df.columns:
            splits_report[split_name] = _label_counts(df[label_col].tolist())
        else:
            splits_report[split_name] = {"total": len(df), "true": 0, "false": 0, "unverified": 0}

        if origin_col not in df.columns:
            continue

        for origin_value, group in df.groupby(origin_col, sort=True):
            origin_name = _normalize_origin_value(origin_value)
            if not origin_name:
                continue

            origin_entry = per_origin.setdefault(
                origin_name,
                {
                    "total": 0,
                    "splits": {},
                },
            )
            origin_entry["total"] += len(group)
            if label_col in group.columns:
                origin_entry["splits"][split_name] = _label_counts(group[label_col].tolist())
            else:
                origin_entry["splits"][split_name] = {
                    "total": len(group),
                    "true": 0,
                    "false": 0,
                    "unverified": 0,
                }

    report = {
        "dataset": hf_dataset,
        "config": hf_config,
        "label_column": label_col,
        "origin_column": origin_col,
        "splits": splits_report,
        "per_origin": per_origin,
        "origin_count": len(per_origin),
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


def _prepare_solver_config(
    *,
    base_config_path: Path,
    output_dir: Path,
    num_rounds_qgen: int,
    max_evidences_per_question: int,
) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to prepare solver config") from exc

    payload = _load_model_config(base_config_path)
    solvers = payload.setdefault("solvers", {})
    retriever = solvers.setdefault("rarr_retriever", {})
    verifier = solvers.setdefault("rarr_verifier", {})

    retriever["num_rounds_qgen"] = max(1, int(num_rounds_qgen))
    verifier["max_evidences_per_question"] = max(1, int(max_evidences_per_question))

    resolved_path = output_dir / "solver_config.resolved.yaml"
    with open(resolved_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return resolved_path


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


def _entry_evidence_count_and_third_party_coverage(entry: dict) -> tuple[int, bool]:
    detail = entry.get("detail", [])
    if not isinstance(detail, list):
        return 0, False

    evidence_count = 0
    has_third_party_coverage = False
    for claim_detail in detail:
        if not isinstance(claim_detail, dict):
            continue
        evidences = claim_detail.get("evidences", [])
        if not isinstance(evidences, list):
            continue
        evidence_count += len(evidences)

        for evidence_item in evidences:
            structural_context = ""
            if isinstance(evidence_item, (list, tuple)) and len(evidence_item) > 2:
                structural_context = str(evidence_item[2] or "").strip()
            elif isinstance(evidence_item, dict):
                structural_context = str(evidence_item.get("structural_context", "") or "").strip()

            if structural_context:
                has_third_party_coverage = True
                break

        if has_third_party_coverage:
            break

    return evidence_count, has_third_party_coverage


def _compute_eval_slice(
    *,
    entries: list[dict],
    labels: list[str] | None,
    prediction_options: list[str],
    gold_options: list[str],
) -> dict:
    matrix = {gold: {pred: 0 for pred in prediction_options} for gold in gold_options}
    rows_scored = 0
    correct = 0
    total_evidence_count = 0
    entries_with_detail = 0

    for idx, entry in enumerate(entries):
        pred_bucket = entry.get("pred_bucket", "unverifiable")
        evidence_count, _ = _entry_evidence_count_and_third_party_coverage(entry)
        total_evidence_count += evidence_count
        if isinstance(entry.get("detail", None), list):
            entries_with_detail += 1

        if labels is None or idx >= len(labels):
            continue

        gold = labels[idx] if labels[idx] in gold_options else "unverified"
        if pred_bucket not in prediction_options:
            pred_bucket = "ambiguous"
        matrix[gold][pred_bucket] += 1
        rows_scored += 1
        if (gold == "true" and pred_bucket == "true") or (gold == "false" and pred_bucket == "false") or (
            gold == "unverified" and pred_bucket == "unverifiable"
        ):
            correct += 1

    return {
        "rows_total": len(entries),
        "rows_scored": rows_scored,
        "accuracy": (correct / rows_scored) if rows_scored else None,
        "avg_evidence_per_row": (total_evidence_count / len(entries) if entries else 0.0),
        "avg_evidence_per_benchmark_query": (total_evidence_count / len(entries) if entries else 0.0),
        "avg_evidence_per_row_with_detail": (
            total_evidence_count / entries_with_detail if entries_with_detail else 0.0
        ),
        "total_evidence_retrieved": total_evidence_count,
        "per_category_prediction_counts": matrix,
    }


def _build_meta_eval(
    *,
    run_id: str,
    dataset_name: str,
    hf_split: str,
    model_id: str,
    model_info: dict,
    condition: str,
    labels: list[str] | None,
    label_audit: dict | None,
    entries: list[dict],
    search_stats: dict | None = None,
) -> dict:
    prediction_options = ["true", "false", "unverifiable", "ambiguous"]
    gold_options = ["true", "false", "unverified"]

    total_metrics = _compute_eval_slice(
        entries=entries,
        labels=labels,
        prediction_options=prediction_options,
        gold_options=gold_options,
    )

    third_party_meta = None
    if condition == "third-party":
        covered_entries = []
        covered_labels = [] if labels is not None else None
        for idx, entry in enumerate(entries):
            _, covered = _entry_evidence_count_and_third_party_coverage(entry)
            if not covered:
                continue
            covered_entries.append(entry)
            if covered_labels is not None and idx < len(labels):
                covered_labels.append(labels[idx])

        covered_metrics = _compute_eval_slice(
            entries=covered_entries,
            labels=covered_labels,
            prediction_options=prediction_options,
            gold_options=gold_options,
        )
        third_party_meta = {
            "rows_with_scored_domain": len(covered_entries),
            "rows_without_scored_domain": len(entries) - len(covered_entries),
            "covered_only_eval": covered_metrics,
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "dataset": dataset_name,
        "split": hf_split,
        "model_id": model_id,
        "model": model_info,
        "rows_total": total_metrics["rows_total"],
        "rows_scored": total_metrics["rows_scored"],
        "accuracy": total_metrics["accuracy"],
        "avg_evidence_per_row": total_metrics["avg_evidence_per_row"],
        "avg_evidence_per_benchmark_query": total_metrics["avg_evidence_per_benchmark_query"],
        "avg_evidence_per_row_with_detail": total_metrics["avg_evidence_per_row_with_detail"],
        "total_evidence_retrieved": total_metrics["total_evidence_retrieved"],
        "search_stats": search_stats or {},
        "per_category_prediction_counts": total_metrics["per_category_prediction_counts"],
        "label_audit": label_audit,
        "third_party": third_party_meta,
    }


def _write_compact_eval_reports(
    *,
    output_dir: Path,
    run_id: str,
    dataset_name: str,
    hf_split: str,
    config_path: Path,
    condition: str,
    labels: list[str] | None,
    label_audit: dict | None,
    entries: list[dict],
    search_stats: dict | None = None,
) -> tuple[Path, Path, dict]:
    model_id, model_info = _model_report_identity(config_path)
    model_path = output_dir / f"{model_id}.json"
    meta_path = output_dir / "meta_eval.json"
    claim_totals = _aggregate_claim_counters(entries)

    model_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "dataset": dataset_name,
        "split": hf_split,
        "model_id": model_id,
        "model": model_info,
        "claims": claim_totals,
        "entries": entries,
    }
    with open(model_path, "w", encoding="utf-8") as handle:
        json.dump(model_payload, handle, indent=2, sort_keys=True)

    meta_payload = _build_meta_eval(
        run_id=run_id,
        dataset_name=dataset_name,
        hf_split=hf_split,
        model_id=model_id,
        model_info=model_info,
        condition=condition,
        labels=labels,
        label_audit=label_audit,
        entries=entries,
        search_stats=search_stats,
    )
    meta_payload["claims"] = claim_totals
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, indent=2, sort_keys=True)

    return model_path, meta_path, meta_payload


def _default_run_id(hf_dataset: str) -> str:
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ("rarr-hf-" if hf_dataset else "rarr-dataset-") + now


def _normalize_origin_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "na", "none", "null", "nan"} else text


def _origin_counts(df: pd.DataFrame, col: str) -> list[tuple[str, int]]:
    normalized = [_normalize_origin_value(v) for v in df[col].tolist()]
    counts: dict[str, int] = {}
    for value in normalized:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))


def _safe_rmtree(path: Path) -> None:
    def _onerror(func, p, exc_info):
        exc = exc_info[1]
        if isinstance(exc, FileNotFoundError):
            return
        raise exc

    shutil.rmtree(path, onerror=_onerror)


def _clip_text(value: object, max_chars: int = 700) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _print_first_sample_sanity(entries: list[dict], labels: list[str] | None) -> None:
    if not entries:
        return

    first = entries[0] if isinstance(entries[0], dict) else {}
    details = first.get("detail") if isinstance(first.get("detail"), list) else []
    if not details:
        print("  sanity first-sample: no verifier details available")
        return

    first_detail = details[0] if isinstance(details[0], dict) else {}
    claim = _clip_text(first_detail.get("claim") or first.get("prompt", ""), max_chars=240)
    evidences = first_detail.get("evidences") if isinstance(first_detail.get("evidences"), list) else []

    first_query = ""
    first_evidence = ""
    first_structural = ""
    if evidences:
        sample = evidences[0]
        if isinstance(sample, (list, tuple)):
            if len(sample) > 0:
                first_query = _clip_text(sample[0], max_chars=240)
            if len(sample) > 1:
                first_evidence = _clip_text(sample[1], max_chars=320)
            if len(sample) > 2:
                first_structural = _clip_text(sample[2], max_chars=280)

    gate_debug = first_detail.get("gate_debug") if isinstance(first_detail.get("gate_debug"), list) else []
    first_gate = gate_debug[0] if gate_debug and isinstance(gate_debug[0], dict) else {}
    final_prompt = _clip_text(first_gate.get("prompt_input", ""), max_chars=1100)
    gate_answer = _clip_text(first_gate.get("raw_response", ""), max_chars=320)
    gate_decision = first_gate.get("decision", "")

    gold_label = first.get("gold_label")
    if gold_label is None and labels:
        gold_label = labels[0]

    print("  sanity first-sample:")
    print(f"    claim: {claim or '<missing>'}")
    print(f"    first query: {first_query or '<missing>'}")
    print(f"    final verifier prompt: {final_prompt or '<missing>'}")
    print(f"    retrieved answer snippet: {first_evidence or '<missing>'}")
    if first_structural:
        print(f"    structural/score context: {first_structural}")
    print(
        f"    verifier answer: {gate_answer or '<empty>'}"
        f" (decision={gate_decision or 'unknown'})"
    )
    print(f"    gold truth: {gold_label if gold_label is not None else '<unavailable>'}")


def run(args: argparse.Namespace) -> None:
    inferred_backend = _configure_backend_environment(args)
    resolved_condition = _configure_condition_environment(args)
    resolved_num_rounds_qgen, resolved_max_evidences = _resolve_runtime_hyperparameters(
        condition=resolved_condition,
        num_rounds_qgen=args.num_rounds_qgen,
        max_evidences_per_question=args.max_evidences_per_question,
    )
    args.num_rounds_qgen = resolved_num_rounds_qgen
    args.max_evidences_per_question = resolved_max_evidences

    if args.hf_list_origin_values and not args.hf_dataset:
        raise ValueError("Use --hf-dataset together with --hf-list-origin-values")

    if not args.hf_list_origin_values:
        _validate_runtime(args)

    if not args.hf_dataset and not args.dataset_path:
        raise ValueError("Provide a dataset path unless you are using --hf-dataset")

    run_id = args.run_id or _default_run_id(args.hf_dataset)
    max_rows = args.max_rows if args.max_rows is not None else _parse_optional_max_rows(args.max_rows_flag)
    max_rows_display = "all" if max_rows is None else str(int(max_rows))

    print(
        "Run settings: "
        f"condition={resolved_condition}, "
        f"backend={inferred_backend}, "
        f"max_rows={max_rows_display}, "
        f"num_rounds_qgen={resolved_num_rounds_qgen}, "
        f"max_evidences_per_question={resolved_max_evidences}"
    )
    output_dir = ROOT_DIR / "eval_results" / "custom" / run_id
    if output_dir.exists():
        _safe_rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_dataset:
        df = _load_hf_dataset(args.hf_dataset, args.hf_config, args.hf_split)

        if args.hf_list_origin_values:
            if args.hf_origin_col not in df.columns:
                raise ValueError(
                    f"Couldn't find origin column '{args.hf_origin_col}' in this Hugging Face split. "
                    f"Available columns: {', '.join(df.columns)}"
                )
            counts = _origin_counts(df, args.hf_origin_col)
            if not counts:
                print(
                    f"No usable origin values were found in column '{args.hf_origin_col}' "
                    f"for {args.hf_dataset}/{args.hf_config}:{args.hf_split}"
                )
                return
            print(f"Origin values in '{args.hf_origin_col}' for {args.hf_dataset}/{args.hf_config}:{args.hf_split}")
            for value, count in counts:
                print(f"{value}\t{count}")
            return

        if args.hf_origin_value:
            if args.hf_origin_col not in df.columns:
                raise ValueError(
                    f"Couldn't find origin column '{args.hf_origin_col}' in this Hugging Face split. "
                    f"Available columns: {', '.join(df.columns)}"
                )
            wanted = args.hf_origin_value.strip().casefold()
            mask = df[args.hf_origin_col].apply(lambda v: _normalize_origin_value(v).casefold() == wanted)
            df = df[mask].reset_index(drop=True)
            if df.empty:
                raise ValueError(
                    f"No rows matched {args.hf_origin_col}='{args.hf_origin_value}' "
                    f"in {args.hf_dataset}/{args.hf_config}:{args.hf_split}"
                )
    else:
        dataset_path = str(Path(args.dataset_path).resolve())
        if not Path(dataset_path).is_file():
            raise FileNotFoundError(f"Couldn't find dataset file: {dataset_path}")
        df = _load_local_dataset(dataset_path)

    if "prompt" not in df.columns:
        if args.hf_dataset and args.hf_claim_col in df.columns:
            df["prompt"] = df[args.hf_claim_col]
        elif "claim" in df.columns:
            df["prompt"] = df["claim"]
        else:
            raise ValueError("Dataset must include a 'prompt' column or a fallback 'claim' column")

    if "response" not in df.columns:
        if args.hf_dataset:
            df["response"] = df["prompt"]
        else:
            raise ValueError("Dataset must include a 'response' column")

    df["prompt"] = df["prompt"].apply(_normalize_text_value)
    df["response"] = df["response"].apply(_normalize_text_value)

    if "source" not in df.columns:
        if args.hf_dataset and args.hf_origin_col in df.columns:
            df["source"] = df[args.hf_origin_col]
        else:
            df["source"] = "hf-dataset" if args.hf_dataset else "custom-dataset"

    df["source"] = df["source"].apply(_normalize_text_value)
    if args.hf_dataset and args.hf_origin_col in df.columns:
        fallback_source = df[args.hf_origin_col].apply(_normalize_text_value)
        df["source"] = df["source"].fillna(fallback_source)
    df["source"] = df["source"].fillna("hf-dataset" if args.hf_dataset else "custom-dataset")

    valid_df = df[df["prompt"].notna() & df["response"].notna()].copy()
    if max_rows is None:
        records = valid_df[["source", "prompt", "response"]].reset_index(drop=True)
    else:
        records = valid_df[["source", "prompt", "response"]].head(max_rows).reset_index(drop=True)
    if records.empty:
        raise ValueError("No usable rows remained after filtering empty prompt/response values")

    labels = None
    label_audit = None
    label_col = _resolve_label_column(valid_df, args.hf_label_col)
    if label_col:
        raw_labels = valid_df[label_col].tolist() if max_rows is None else valid_df[label_col].head(max_rows).tolist()
        labels = [_normalize_label(v) for v in raw_labels]
        label_audit = _build_label_audit(raw_labels, labels)

    dataset_name = str(df["source"].iloc[0]) if not df.empty else (args.hf_dataset or "custom-dataset")
    if args.hf_dataset and args.hf_origin_value:
        dataset_name = args.hf_origin_value

    _python_path_setup()
    from context_core.benchmark import evaluate_free_text_with_auto_checker
    from context_core.utils import search as search_utils

    search_utils.reset_search_stats()

    base_config_path = (LOCAL_CORE_DIR / "config" / "rarr_web_service_config.yaml").resolve()
    resolved_solver_config = _prepare_solver_config(
        base_config_path=base_config_path,
        output_dir=output_dir,
        num_rounds_qgen=resolved_num_rounds_qgen,
        max_evidences_per_question=resolved_max_evidences,
    )

    solver_args = argparse.Namespace(
        user_src=str((LOCAL_CORE_DIR / "solvers").resolve()),
        config=str(resolved_solver_config.resolve()),
        output=str(output_dir.resolve()),
        persist_outputs=False,
        openai_apikey=os.getenv("OPENAI_API_KEY"),
        factcheck_model=args.factcheck_model,
        rarr_model=args.rarr_model,
    )

    entries = evaluate_free_text_with_auto_checker(
        records.to_dict("records"),
        response_column_name="dataset_response",
        args=solver_args,
        projectdir=str(output_dir.resolve()),
    )

    _annotate_eval_entries(entries)
    if labels is not None:
        for idx, gold in enumerate(labels):
            if idx < len(entries):
                entries[idx]["gold_label"] = gold

    model_path, meta_path, meta_payload = _write_compact_eval_reports(
        output_dir=output_dir,
        run_id=run_id,
        dataset_name=args.hf_dataset or dataset_name,
        hf_split=args.hf_split,
        config_path=Path(solver_args.config),
        condition=resolved_condition,
        labels=labels,
        label_audit=label_audit,
        entries=entries,
        search_stats=search_utils.get_search_stats(),
    )

    print(f"Run finished: {run_id}")
    print(f"Rows processed: {len(records)} | Output folder: {output_dir}")
    print(f"Model log: {model_path} | Summary report: {meta_path}")

    if args.hf_dataset:
        stats_path, _ = _write_dataset_split_stats(
            args.hf_dataset,
            args.hf_config,
            args.hf_label_col,
            args.hf_origin_col,
        )
        print(f"Dataset split stats saved to: {stats_path}")

    ss = meta_payload.get("search_stats", {})
    print(
        f"Run summary:\n"
        f"  rows scored: {meta_payload['rows_scored']}\n"
        f"  avg evidence/row: {meta_payload['avg_evidence_per_row']:.2f}\n"
        f"  avg evidence/benchmark query: {meta_payload['avg_evidence_per_benchmark_query']:.2f}\n"
        f"  avg evidence/row (with detail): {meta_payload['avg_evidence_per_row_with_detail']:.2f}\n"
        f"  total evidence retrieved: {meta_payload['total_evidence_retrieved']}\n"
        f"  search timeouts: {ss.get('search_timeouts', 0)}\n"
        f"  serper->ddg fallbacks: {ss.get('serper_timeout_fallback_to_ddg', 0)}\n"
        f"  provider failures: {ss.get('provider_failures', 0)}\n"
        f"  queries with no results: {ss.get('queries_with_no_results', 0)}"
    )
    tp_total = ss.get("third_party_evidence_total", 0)
    tp_scored = ss.get("third_party_evidence_scored", 0)
    if tp_total > 0:
        print(f"  third-party score coverage: {tp_scored}/{tp_total} ({100.0 * tp_scored / tp_total:.1f}%)")
    tp_meta = meta_payload.get("third_party") or {}
    covered_eval = tp_meta.get("covered_only_eval") if isinstance(tp_meta, dict) else None
    if covered_eval:
        print(
            "  third-party covered-only eval: "
            f"rows={covered_eval.get('rows_total', 0)} "
            f"(covered={tp_meta.get('rows_with_scored_domain', 0)}, "
            f"uncovered={tp_meta.get('rows_without_scored_domain', 0)}), "
            f"accuracy={covered_eval.get('accuracy')}"
        )

    if labels is not None:
        print(f"  accuracy: {meta_payload['accuracy']}")
        audit = meta_payload.get("label_audit") or {}
        if audit:
            norm = audit.get("normalized_counts", {})
            print(
                f"  label audit: rows={audit.get('rows_considered', 0)}, raw_unique={audit.get('raw_unique', 0)}, "
                f"fallback={audit.get('fallback_count', 0)}, normalized(true={norm.get('true', 0)}, "
                f"false={norm.get('false', 0)}, unverified={norm.get('unverified', 0)})"
            )
            top_fallback = audit.get("top_fallback_labels", [])
            if top_fallback:
                fallback_text = ", ".join(
                    f"{item.get('label')}={item.get('count')}" for item in top_fallback[:5]
                )
                print(f"  label audit fallback examples: {fallback_text}")
        for category in ("true", "false", "unverified"):
            counts = meta_payload["per_category_prediction_counts"].get(category, {})
            print(
                f"  {category}: true={counts.get('true', 0)}, false={counts.get('false', 0)}, "
                f"unverifiable={counts.get('unverifiable', 0)}, ambiguous={counts.get('ambiguous', 0)}"
            )
    else:
        if args.hf_dataset:
            print(f"No usable label column was found (expected '{args.hf_label_col}'), so ground-truth metrics were skipped.")
        else:
            print("No usable label column was found in this local dataset, so ground-truth metrics were skipped.")

    _print_first_sample_sanity(entries, labels)


def main() -> None:
    args = parse_args()
    if args.show_prompts:
        _show_prompts()
        return
    run(args)


if __name__ == "__main__":
    main()
