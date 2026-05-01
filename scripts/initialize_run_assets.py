from __future__ import annotations

import argparse
import fcntl
import json
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
INVALID_TEXT_VALUES = {"", "na", "none", "null", "nan", "n/a"}
REQUIRED_SPLIT_COLUMNS = ("prompt", "response")
DEFAULT_MIN_SAMPLES_PER_SPLIT = 20
DEFAULT_MAX_SAMPLES_PER_LABEL = 5000
ALLOWED_ORIGIN_SLUGS = {
    "fakecovid",
    "coaid",
    "ct-fan",
    "defakts",
    "liar",
    "isot-fake-news",
    "rumors",
    "claimskg",
    "benjamin-political-news",
}
DEFAULT_ORIGIN_SLUG = "liar"
PROMPT_FALLBACK_COLUMNS = (
    "prompt",
    "claim",
    "tweet_text",
    "social_media_text",
    "article_title",
    "article_headline",
    "summary",
    "text",
    "content",
    "question",
)
RESPONSE_FALLBACK_COLUMNS = (
    "response",
    "answer",
    "what's true",
    "what's false",
    "what's unknown",
)


def _normalize_origin(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return ""
    text = str(value).strip()
    if text.lower() in INVALID_TEXT_VALUES:
        return ""
    return text


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    if text.lower() in INVALID_TEXT_VALUES:
        return None
    return text


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower().strip()).strip("-") or "unknown"


def _is_allowed_origin(origin: str) -> bool:
    return _slugify(origin) in ALLOWED_ORIGIN_SLUGS


def _split_counts(total: int) -> tuple[int, int, int]:
    train = int(total * 0.6)
    val = int(total * 0.2)
    test = total - train - val
    if total > 0 and test == 0:
        if val > 1:
            val -= 1
        elif train > 1:
            train -= 1
        test = total - train - val
    return train, val, test


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


def _stratified_split_by_label(group: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for idx, (_, label_df) in enumerate(group.groupby("__label_norm", sort=True)):
        label_df = label_df.sample(frac=1.0, random_state=seed + idx).reset_index(drop=True)
        train_n, val_n, _ = _split_counts(len(label_df))
        train_parts.append(label_df.iloc[:train_n].copy())
        val_parts.append(label_df.iloc[train_n : train_n + val_n].copy())
        test_parts.append(label_df.iloc[train_n + val_n :].copy())

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else group.iloc[:0].copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else group.iloc[:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else group.iloc[:0].copy()

    if not train_df.empty:
        train_df = train_df.sample(frac=1.0, random_state=seed + 101).reset_index(drop=True)
    if not val_df.empty:
        val_df = val_df.sample(frac=1.0, random_state=seed + 202).reset_index(drop=True)
    if not test_df.empty:
        test_df = test_df.sample(frac=1.0, random_state=seed + 303).reset_index(drop=True)

    return train_df, val_df, test_df


def _cap_group_labels(group: pd.DataFrame, seed: int, max_samples_per_label: int) -> pd.DataFrame:
    if "__label_norm" not in group.columns:
        return group

    capped_parts: list[pd.DataFrame] = []
    for idx, (_, label_df) in enumerate(group.groupby("__label_norm", sort=True)):
        if len(label_df) > max_samples_per_label:
            label_df = label_df.sample(n=max_samples_per_label, random_state=seed + idx)
        capped_parts.append(label_df)

    if not capped_parts:
        return group.iloc[:0].copy()

    capped = pd.concat(capped_parts, ignore_index=True)
    return capped.sample(frac=1.0, random_state=seed + 404).reset_index(drop=True)


def _load_all_hf_rows(repo: str, config: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required. Install it in ctxt-env first.") from exc

    attempts = [
        {"path": repo, "name": config},
        {"path": repo, "name": config, "revision": "main"},
        {
            "path": repo,
            "name": config,
            "revision": "main",
            "download_mode": "force_redownload",
        },
    ]

    last_exc: Exception | None = None
    ds = None
    for idx, kwargs in enumerate(attempts, start=1):
        try:
            ds = load_dataset(**kwargs)
            break
        except Exception as exc:  # pragma: no cover - depends on remote HF state
            last_exc = exc
            if idx < len(attempts):
                print(
                    f"[init] HF load attempt {idx}/{len(attempts)} failed: {exc}. Retrying ..."
                )

    if ds is None:
        raise RuntimeError(
            f"Failed to load HF dataset {repo}/{config} after {len(attempts)} attempts"
        ) from last_exc

    frames = []
    for split_name, split_ds in ds.items():
        frame = split_ds.to_pandas()
        frame["__original_hf_split"] = split_name
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No dataset splits were found for {repo}/{config}")
    return pd.concat(frames, ignore_index=True)


def _resolve_origin_column(df: pd.DataFrame, configured: str) -> str:
    candidates = [configured, "source", "dataset", "source_dataset"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Couldn't find an origin column. Tried: {', '.join(candidates)}. Available columns: {', '.join(df.columns)}"
    )


def _write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=True)


def _asset_lock_path(metadata_path: Path, shards_dir: Path) -> Path:
    shared_root = metadata_path.resolve().parent
    try:
        if shards_dir.resolve().is_relative_to(ROOT_DIR.resolve()):
            shared_root = ROOT_DIR.resolve()
    except FileNotFoundError:
        pass
    return shared_root / ".initialize_run_assets.lock"


@contextmanager
def _locked_asset_init(metadata_path: Path, shards_dir: Path):
    lock_path = _asset_lock_path(metadata_path, shards_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"[init] Another job is already refreshing shared assets. Waiting for {lock_path} ...")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_rmtree(path: Path) -> None:
    def _onerror(func, p, exc_info):
        exc = exc_info[1]
        if isinstance(exc, FileNotFoundError):
            return
        raise exc

    shutil.rmtree(path, onerror=_onerror)


def _prepare_rows(df: pd.DataFrame, origin_col: str) -> tuple[pd.DataFrame, int]:
    prepared = df.copy()
    prepared["__origin"] = prepared[origin_col].apply(_normalize_origin)

    prompt_cols = [col for col in PROMPT_FALLBACK_COLUMNS if col in prepared.columns]
    if not prompt_cols:
        raise ValueError(
            "Dataset needs at least one usable prompt text column "
            f"(checked: {', '.join(PROMPT_FALLBACK_COLUMNS)})"
        )
    prompt_candidates = prepared[prompt_cols].apply(lambda col: col.apply(_normalize_text))
    prepared["prompt"] = prompt_candidates.bfill(axis=1).iloc[:, 0]

    response_cols = [col for col in RESPONSE_FALLBACK_COLUMNS if col in prepared.columns]
    if response_cols:
        response_candidates = prepared[response_cols].apply(lambda col: col.apply(_normalize_text))
        prepared["response"] = response_candidates.bfill(axis=1).iloc[:, 0].fillna(prepared["prompt"])
    else:
        prepared["response"] = prepared["prompt"]

    prepared["prompt"] = prepared["prompt"].apply(_normalize_text)
    prepared["response"] = prepared["response"].apply(_normalize_text)
    prepared["source"] = prepared["__origin"]

    valid_mask = (
        prepared["__origin"].ne("")
        & prepared["prompt"].notna()
        & prepared["response"].notna()
    )
    filtered_out = int((~valid_mask).sum())
    prepared = prepared.loc[valid_mask].copy()
    return prepared, filtered_out


def _is_usable_split_frame(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    if any(col not in frame.columns for col in REQUIRED_SPLIT_COLUMNS):
        return False
    for col in REQUIRED_SPLIT_COLUMNS:
        values = frame[col].apply(_normalize_text)
        if values.isna().any():
            return False
    return True


def _has_stable_split_sizes(
    train_n: int,
    val_n: int,
    test_n: int,
    *,
    min_samples_per_split: int,
) -> bool:
    return min(train_n, val_n, test_n) >= max(1, int(min_samples_per_split))


def _initialize_local_splits(
    *,
    data_dir: Path,
    metadata_path: Path,
    hf_dataset: str,
    hf_config: str,
    origin_col: str,
    label_col: str,
    seed: int,
    min_samples_per_split: int,
    max_samples_per_label: int,
    force: bool,
) -> None:
    if metadata_path.exists() and not force:
        print(f"[init] Using existing split metadata at {metadata_path}")
        return

    print(f"[init] Loading HF dataset {hf_dataset}/{hf_config} (all splits)")
    df = _load_all_hf_rows(hf_dataset, hf_config)
    effective_origin_col = _resolve_origin_column(df, origin_col)
    original_row_count = len(df)
    df, filtered_row_count = _prepare_rows(df, effective_origin_col)
    if df.empty:
        raise RuntimeError("No usable rows remained after cleaning origin values")

    effective_label_col = label_col if label_col in df.columns else None
    if effective_label_col is None and "label" in df.columns:
        effective_label_col = "label"
    if effective_label_col is not None:
        df["__label_norm"] = df[effective_label_col].apply(_normalize_label)
    else:
        print(
            "[init] Warning: no label column found, so per-origin splits will be random rather than stratified"
        )

    if force:
        for child in data_dir.iterdir() if data_dir.exists() else []:
            if child.name in {"domain_ratings.csv", "metadata.json"}:
                continue
            if child.is_dir():
                for marker in (child / "train", child / "val", child / "test"):
                    if marker.exists():
                        _safe_rmtree(child)
                        break
            elif child.exists():
                child.unlink(missing_ok=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    datasets_meta: dict[str, dict[str, object]] = {}
    skipped_origins: list[dict[str, object]] = []

    origin_groups = df.groupby("__origin", sort=True)
    total_origins = int(df["__origin"].nunique(dropna=True))
    for origin, group in tqdm(
        origin_groups,
        total=total_origins,
        desc="[init] Building per-origin splits",
        unit="origin",
    ):
        if not _is_allowed_origin(origin):
            skipped_origins.append(
                {
                    "origin": origin,
                    "reason": "origin_not_allowlisted",
                }
            )
            continue

        group = group.reset_index(drop=True)
        if "__label_norm" in group.columns:
            group = _cap_group_labels(
                group,
                seed=seed,
                max_samples_per_label=max(1, int(max_samples_per_label)),
            )
        total = len(group)
        if "__label_norm" in group.columns:
            train_df, val_df, test_df = _stratified_split_by_label(group, seed=seed)
        else:
            shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            train_n, val_n, _ = _split_counts(total)
            train_df = shuffled.iloc[:train_n].copy()
            val_df = shuffled.iloc[train_n : train_n + val_n].copy()
            test_df = shuffled.iloc[train_n + val_n :].copy()

        for frame in (train_df, val_df, test_df):
            for helper_col in ("__origin", "__label_norm"):
                if helper_col in frame.columns:
                    frame.drop(columns=[helper_col], inplace=True)

        train_n = len(train_df)
        val_n = len(val_df)
        test_n = len(test_df)

        if not _has_stable_split_sizes(
            train_n,
            val_n,
            test_n,
            min_samples_per_split=min_samples_per_split,
        ):
            skipped_origins.append(
                {
                    "origin": origin,
                    "total": total,
                    "train": train_n,
                    "val": val_n,
                    "test": test_n,
                    "reason": "insufficient_split_size",
                    "min_samples_per_split": min_samples_per_split,
                }
            )
            continue

        if not all(_is_usable_split_frame(frame) for frame in (train_df, val_df, test_df)):
            skipped_origins.append(
                {
                    "origin": origin,
                    "total": total,
                    "train": train_n,
                    "val": val_n,
                    "test": test_n,
                    "reason": "malformed_required_columns",
                }
            )
            continue

        slug = _slugify(origin)
        base = data_dir / slug
        train_path = base / "train" / "data.jsonl"
        val_path = base / "val" / "data.jsonl"
        test_path = base / "test" / "data.jsonl"

        _write_jsonl(train_df, train_path)
        _write_jsonl(val_df, val_path)
        _write_jsonl(test_df, test_path)

        datasets_meta[slug] = {
            "origin": origin,
            "total": total,
            "train": train_n,
            "val": val_n,
            "test": test_n,
            "label_distribution": {
                "train": train_df[effective_label_col].apply(_normalize_label).value_counts().to_dict()
                if effective_label_col and effective_label_col in train_df.columns
                else {},
                "val": val_df[effective_label_col].apply(_normalize_label).value_counts().to_dict()
                if effective_label_col and effective_label_col in val_df.columns
                else {},
                "test": test_df[effective_label_col].apply(_normalize_label).value_counts().to_dict()
                if effective_label_col and effective_label_col in test_df.columns
                else {},
            },
            "paths": {
                "train": str(train_path.resolve()),
                "val": str(val_path.resolve()),
                "test": str(test_path.resolve()),
            },
        }

    if not datasets_meta:
        raise RuntimeError("No usable per-origin datasets were created")

    if DEFAULT_ORIGIN_SLUG in datasets_meta:
        default_slug = DEFAULT_ORIGIN_SLUG
        default_info = datasets_meta[default_slug]
    else:
        default_slug, default_info = min(
            datasets_meta.items(),
            key=lambda item: (int(item[1].get("total", 0)), item[0]),
        )
    default_split = "test" if int(default_info.get("test", 0)) > 0 else "train"

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "origin_column": effective_origin_col,
        "label_column": effective_label_col,
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "stratified_by_label": bool(effective_label_col),
        "stable_splits_only": True,
        "min_samples_per_split": min_samples_per_split,
        "max_samples_per_label": max_samples_per_label,
        "allowed_origin_slugs": sorted(ALLOWED_ORIGIN_SLUGS),
        "default_origin_slug": DEFAULT_ORIGIN_SLUG,
        "rows_loaded": original_row_count,
        "rows_retained": len(df),
        "rows_filtered": filtered_row_count,
        "skipped_unstable_origins": skipped_origins,
        "datasets": datasets_meta,
        "default_dataset": {
            "slug": default_slug,
            "origin": default_info.get("origin", default_slug),
            "split": default_split,
            "path": default_info.get("paths", {}).get(default_split, ""),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[init] Saved split metadata to {metadata_path}")
    print(f"[init] Generated per-origin datasets: {len(datasets_meta)}")
    if skipped_origins:
        print(f"[init] Skipped origins that were too small or malformed: {len(skipped_origins)}")


def _initialize_structural_shards(
    *,
    shards_dir: Path,
    shard_count: int,
    hook_path: Path,
    month: str,
    edges_url: str,
    shard_workers: int,
    force: bool,
) -> None:
    meta_path = shards_dir / "_meta.json"
    if meta_path.exists() and not force:
        print(f"[init] Serving shards already exist at {meta_path}")
        return

    if not hook_path.is_file():
        raise FileNotFoundError(f"Couldn't find the structural hook at {hook_path}")

    shards_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(hook_path.resolve()),
        "--build-serving-shards",
        str(shards_dir.resolve()),
        "--month-file",
        f"{month}={edges_url}",
        "--serving-shard-count",
        str(shard_count),
        "--shard-workers",
        str(shard_workers),
        "--verbose",
    ]
    if force:
        cmd.append("--force-rebuild-serving-shards")

    print("[init] Building structural serving shards")
    subprocess.run(cmd, check=True)
    if not meta_path.exists():
        raise RuntimeError(f"Shard build finished, but no metadata file was written at {meta_path}")
    print(f"[init] Structural hook is ready: {hook_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize local per-origin splits and structural hook shards for run.sh defaults."
    )
    parser.add_argument("--hf-dataset", default="ComplexDataLab/Misinfo_Datasets")
    parser.add_argument("--hf-config", default="default")
    parser.add_argument("--origin-col", default="source")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default=str((ROOT_DIR / "data").resolve()))
    parser.add_argument("--metadata-path", default=str((ROOT_DIR / "data" / "metadata.json").resolve()))
    parser.add_argument("--structural-shards-dir", required=True)
    parser.add_argument("--structural-shard-count", type=int, default=256)
    parser.add_argument("--hook-path", default=str((ROOT_DIR / "scripts" / "hook.py").resolve()))
    parser.add_argument("--edges-month", default="mar2025")
    parser.add_argument(
        "--edges-url",
        default="https://huggingface.co/datasets/credi-net/CrediBench/resolve/main/march2025/edges.csv.gz",
    )
    parser.add_argument("--shard-workers", type=int, default=1)
    parser.add_argument("--min-samples-per-split", type=int, default=DEFAULT_MIN_SAMPLES_PER_SPLIT)
    parser.add_argument("--max-samples-per-label", type=int, default=DEFAULT_MAX_SAMPLES_PER_LABEL)
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--force-shards", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    force_splits = args.force or args.force_splits
    force_shards = args.force or args.force_shards

    metadata_path = Path(args.metadata_path)
    shards_dir = Path(args.structural_shards_dir)

    with _locked_asset_init(metadata_path, shards_dir):
        _initialize_local_splits(
            data_dir=Path(args.data_dir),
            metadata_path=metadata_path,
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            origin_col=args.origin_col,
            label_col=args.label_col,
            seed=args.seed,
            min_samples_per_split=max(1, int(args.min_samples_per_split)),
            max_samples_per_label=max(1, int(args.max_samples_per_label)),
            force=force_splits,
        )
        _initialize_structural_shards(
            shards_dir=shards_dir,
            shard_count=args.structural_shard_count,
            hook_path=Path(args.hook_path),
            month=args.edges_month,
            edges_url=args.edges_url,
            shard_workers=args.shard_workers,
            force=force_shards,
        )


if __name__ == "__main__":
    main()
