from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
import os
import re
import shutil
import ssl
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urlparse
from urllib.request import urlopen

logger = logging.getLogger(__name__)

DEFAULT_SERVING_SHARD_COUNT = 256
_SHARD_BUILD_LOCK_FILE = "_build.lock"
_SHARD_BUILD_TMP_DIR = "_build_tmp"
_SHARD_BUILD_PROGRESS_FILE = "_progress.json"
DEFAULT_QUERY_MODE = "latest"
DEFAULT_HOPS = 2
DEFAULT_MAX_DOMAINS_PER_HOP = 10
SERVING_SHARD_META_FILE = "_meta.json"

MONTHS: tuple[str, ...] = (
    "oct2024",
    "nov2024",
    "dec2024",
    "jan2025",
    "feb2025",
    "mar2025",
    "apr2025",
)

CREDIBENCH_REPO_ID = "credi-net/CrediBench"
_CREDIBENCH_MONTH_FOLDERS = {
    "oct2024": "oct2024",
    "nov2024": "nov2024",
    "dec2024": "dec2024",
    "jan2025": "jan2025",
    "feb2025": "feb2025",
    "mar2025": "march2025",
    "apr2025": "april2025",
}

_MONTH_ALIASES = {
    "oct2024": "oct2024",
    "october2024": "oct2024",
    "nov2024": "nov2024",
    "november2024": "nov2024",
    "dec2024": "dec2024",
    "december2024": "dec2024",
    "jan2025": "jan2025",
    "january2025": "jan2025",
    "feb2025": "feb2025",
    "february2025": "feb2025",
    "mar2025": "mar2025",
    "march2025": "mar2025",
    "apr2025": "apr2025",
    "april2025": "apr2025",
}


def canonical_month(month: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", month.lower())
    if key not in _MONTH_ALIASES:
        raise ValueError(
            f"Unsupported month label '{month}'. Expected one of: {', '.join(MONTHS)}"
        )
    return _MONTH_ALIASES[key]


def normalize_month_paths(month_to_file: Mapping[str, str | Path]) -> dict[str, str | Path]:
    normalized: dict[str, str | Path] = {}
    for month, file_path in month_to_file.items():
        normalized[canonical_month(month)] = _normalize_source(file_path)
    return normalized


def _reverse_domain(domain: str) -> str:
    parts = domain.strip().lower().split(".")
    return ".".join(reversed(parts))


def _domain_variants(domain: str) -> list[str]:
    normalized = domain.strip().lower()
    reversed_domain = _reverse_domain(normalized)
    if reversed_domain == normalized:
        return [normalized]
    return [normalized, reversed_domain]


def _normalize_source(source: str | Path) -> str | Path:
    if isinstance(source, Path):
        return source
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return source
    return Path(source)


def _is_remote_source(source: str | Path) -> bool:
    return isinstance(source, str) and urlparse(source).scheme in {"http", "https"}


def _huggingface_dataset_url(repo_id: str, revision: str, relative_path: str) -> str:
    repo_prefix = quote(repo_id, safe="/")
    revision_part = quote(revision, safe="")
    path_part = quote(relative_path, safe="/")
    return (
        f"https://huggingface.co/datasets/{repo_prefix}/resolve/"
        f"{revision_part}/{path_part}"
    )


def credibench_month_urls(
    *,
    repo_id: str = CREDIBENCH_REPO_ID,
    revision: str = "main",
    months: tuple[str, ...] = MONTHS,
) -> dict[str, str]:
    return {
        month: _huggingface_dataset_url(
            repo_id,
            revision,
            f"{_CREDIBENCH_MONTH_FOLDERS[month]}/edges.csv.gz",
        )
        for month in months
    }


def parse_month_file_args(month_file_args: list[str]) -> dict[str, str | Path]:
    mapping: dict[str, str | Path] = {}
    for item in month_file_args:
        if "=" not in item:
            raise ValueError(f"Invalid --month-file value: {item!r}. Expected month=path")
        month, raw_path = item.split("=", 1)
        mapping[canonical_month(month)] = _normalize_source(raw_path)
    return mapping


def _open_source_text_reader(source: str | Path):
    is_remote = _is_remote_source(source)
    if is_remote:
        url = str(source)
        context = None
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            context = None

        try:
            raw = urlopen(url, context=context)
        except URLError:
            if os.getenv("HOOK_ALLOW_INSECURE_SSL", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                logger.warning(
                    "SSL verification failed for %s; retrying without certificate verification because HOOK_ALLOW_INSECURE_SSL is enabled",
                    url,
                )
                raw = urlopen(url, context=ssl._create_unverified_context())
            else:
                raise
    else:
        raw = open(str(source), "rb")

    try:
        path = urlparse(str(source)).path.lower() if is_remote else str(source).lower()
        if path.endswith(".gz"):
            gz = gzip.GzipFile(fileobj=raw)
            return io.TextIOWrapper(gz, encoding="utf-8", errors="ignore", newline="")
        return io.TextIOWrapper(raw, encoding="utf-8", errors="ignore", newline="")
    except Exception:
        raw.close()
        raise


def _iter_edges_pyarrow_chunked(
    source: Path,
    src_col: str,
    dst_col: str,
    csv_delimiter: str,
):
    try:
        import pyarrow as pa
        import pyarrow.csv as pacsv
    except ImportError as exc:
        raise RuntimeError("pyarrow is not installed") from exc

    block_size_default = 8 * 1024 * 1024
    try:
        block_size = max(1, int(os.getenv("HOOK_ARROW_BLOCK_SIZE", str(block_size_default))))
    except ValueError:
        block_size = block_size_default

    base_stream = pa.input_stream(str(source))
    compressed_stream = None
    stream = base_stream
    if str(source).lower().endswith(".gz"):
        compressed_stream = pa.CompressedInputStream(base_stream, "gzip")
        stream = compressed_stream

    try:
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(
                use_threads=True,
                block_size=block_size,
                autogenerate_column_names=False,
            ),
            parse_options=pacsv.ParseOptions(delimiter=csv_delimiter),
            convert_options=pacsv.ConvertOptions(
                include_columns=[src_col, dst_col],
                column_types={src_col: pa.string(), dst_col: pa.string()},
                strings_can_be_null=True,
            ),
        )
        if src_col not in reader.schema.names or dst_col not in reader.schema.names:
            raise ValueError(
                f"Expected columns '{src_col}' and '{dst_col}' in {source}; got {reader.schema.names}"
            )

        src_idx = reader.schema.get_field_index(src_col)
        dst_idx = reader.schema.get_field_index(dst_col)
        while True:
            try:
                batch = reader.read_next_batch()
            except StopIteration:
                break

            src_values = batch.column(src_idx).to_pylist()
            dst_values = batch.column(dst_idx).to_pylist()
            for src, dst in zip(src_values, dst_values):
                src_norm = str(src).strip().lower() if src is not None else ""
                dst_norm = str(dst).strip().lower() if dst is not None else ""
                if not src_norm or not dst_norm:
                    continue
                yield src_norm, dst_norm
    finally:
        try:
            if compressed_stream is not None:
                compressed_stream.close()
        finally:
            base_stream.close()


def _iter_edges(
    source: str | Path,
    src_col: str,
    dst_col: str,
    csv_delimiter: str,
    csv_has_header: bool,
):
    if not csv_has_header:
        raise ValueError("CSV header is required")

    if isinstance(source, Path):
        try:
            yield from _iter_edges_pyarrow_chunked(source, src_col, dst_col, csv_delimiter)
            return
        except Exception as exc:
            logger.warning(
                "[edges] pyarrow chunked reader unavailable for %s (%s); falling back to Python csv reader",
                source,
                exc,
            )

    reader_handle = _open_source_text_reader(source)
    try:
        reader = csv.DictReader(reader_handle, delimiter=csv_delimiter)
        if not reader.fieldnames or src_col not in reader.fieldnames or dst_col not in reader.fieldnames:
            raise ValueError(
                f"Expected columns '{src_col}' and '{dst_col}' in {source}; got {reader.fieldnames}"
            )
        try:
            for row in reader:
                src = str(row.get(src_col, "")).strip().lower()
                dst = str(row.get(dst_col, "")).strip().lower()
                if not src or not dst:
                    continue
                yield src, dst
        except EOFError:
            logger.warning("[edges] truncated gzip in %s — partial data used", source)
    finally:
        reader_handle.close()


def _latest_month_for_map(month_map: Mapping[str, str | Path] | None) -> str:
    if not month_map:
        return MONTHS[-1]
    for month in reversed(MONTHS):
        if month in month_map:
            return month
    return MONTHS[-1]


def one_hop_neighbors_by_month(
    domain: str,
    month_to_file: Mapping[str, str | Path],
    *,
    src_col: str = "src",
    dst_col: str = "dst",
    csv_delimiter: str = ",",
    csv_has_header: bool = True,
) -> dict[str, list[str]]:
    if not domain or not domain.strip():
        raise ValueError("domain must be a non-empty string")

    month_paths = normalize_month_paths(month_to_file)
    variants = set(_domain_variants(domain))
    result: dict[str, list[str]] = {month: [] for month in MONTHS}

    for month in MONTHS:
        source = month_paths.get(month)
        if source is None:
            continue
        if isinstance(source, Path) and not source.exists():
            continue
        neighbors: set[str] = set()
        for src, dst in _iter_edges(source, src_col, dst_col, csv_delimiter, csv_has_header):
            if src in variants and dst not in variants:
                neighbors.add(dst)
            elif dst in variants and src not in variants:
                neighbors.add(src)
        result[month] = sorted(neighbors)

    return result


def batch_one_hop_neighbors_by_month(
    domains: list[str],
    month_to_file: Mapping[str, str | Path],
    *,
    src_col: str = "src",
    dst_col: str = "dst",
    csv_delimiter: str = ",",
    csv_has_header: bool = True,
    months: list[str] | None = None,
    max_neighbors_per_month: int = 0,
) -> dict[str, dict[str, list[str]]]:
    input_domains = sorted({d.strip().lower() for d in domains if d.strip()})
    if not input_domains:
        raise ValueError("domains must contain at least one non-empty domain")

    month_paths = normalize_month_paths(month_to_file)
    variant_sets = {domain: set(_domain_variants(domain)) for domain in input_domains}
    month_plan = months if months else list(MONTHS)
    month_lookup = set(month_plan)

    result_sets: dict[str, dict[str, set[str]]] = {
        domain: {month: set() for month in month_plan}
        for domain in input_domains
    }

    for month in month_plan:
        source = month_paths.get(month)
        if source is None:
            continue
        if isinstance(source, Path) and not source.exists():
            continue

        for src, dst in _iter_edges(source, src_col, dst_col, csv_delimiter, csv_has_header):
            for domain in input_domains:
                variants = variant_sets[domain]
                if src in variants and dst not in variants:
                    if max_neighbors_per_month <= 0 or len(result_sets[domain][month]) < max_neighbors_per_month:
                        result_sets[domain][month].add(dst)
                elif dst in variants and src not in variants:
                    if max_neighbors_per_month <= 0 or len(result_sets[domain][month]) < max_neighbors_per_month:
                        result_sets[domain][month].add(src)

    return {
        domain: {month: sorted(result_sets[domain][month]) for month in month_plan}
        for domain in input_domains
    }

def _stable_shard_id(domain: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    digest = hashlib.blake2b(domain.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % shard_count


def _acquire_shard_build_lock(shards_dir: Path):
    lock_path = shards_dir / _SHARD_BUILD_LOCK_FILE
    lock_fd = open(lock_path, "w")
    try:
        import fcntl
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock_fd.close()
            raise RuntimeError(
                f"Another shard build is already running in {shards_dir} "
                f"(lock: {lock_path}). Kill that job or use --force-rebuild-serving-shards."
            )
    except ImportError:
        pass  # non-Linux: skip locking
    lock_fd.write(f"{os.getpid()}\n")
    lock_fd.flush()
    return lock_fd


def _release_shard_build_lock(lock_fd) -> None:
    if lock_fd is None:
        return
    try:
        import fcntl
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock_fd.close()
    except Exception:
        pass


def _process_month_to_tmp(
    month: str,
    source: str | Path,
    build_tmp: Path,
    shard_count: int,
    src_col: str,
    dst_col: str,
    csv_delimiter: str,
    csv_has_header: bool,
    show_progress: bool,
    log_progress: bool,
) -> int:
    """Stream one month's edges into per-shard temp gzip files. Returns directed edge count."""
    month_tmp = build_tmp / f"month={month}"
    if month_tmp.exists():
        shutil.rmtree(month_tmp)
    month_tmp.mkdir(parents=True)

    handles: list = []
    writers: list = []
    for shard_id in range(shard_count):
        h = gzip.open(month_tmp / f"shard_{shard_id}.csv.gz", "wt", encoding="utf-8", newline="")
        handles.append(h)
        writers.append(csv.writer(h))

    total = 0
    try:
        if log_progress:
            logger.info("[shards] ingesting month=%s from %s", month, source)

        row_bar = None
        if show_progress:
            try:
                from tqdm import tqdm as _tqdm  # type: ignore
                row_bar = _tqdm(desc=f"{month} rows", unit="row", leave=False)
            except Exception:
                pass

        for src, dst in _iter_edges(source, src_col, dst_col, csv_delimiter, csv_has_header):
            if src == dst:
                continue
            for domain, neighbor in ((src, dst), (dst, src)):
                writers[_stable_shard_id(domain, shard_count)].writerow([domain, month, neighbor])
            total += 1
            if row_bar is not None:
                row_bar.update(1)

        if row_bar is not None:
            row_bar.close()
    finally:
        for h in handles:
            try:
                h.close()
            except Exception:
                pass

    return total


def _merge_month_into_shards(
    month: str,
    build_tmp: Path,
    shards_dir: Path,
    shard_count: int,
    log_progress: bool,
) -> None:
    month_tmp = build_tmp / f"month={month}"
    if not month_tmp.exists():
        return
    if log_progress:
        logger.info("[shards] merging month=%s into final shards", month)
    for shard_id in range(shard_count):
        src = month_tmp / f"shard_{shard_id}.csv.gz"
        dst = shards_dir / f"shard_id={shard_id}" / "neighbors.csv.gz"
        if not src.exists():
            continue
        with open(dst, "ab") as out_f, open(src, "rb") as in_f:
            out_f.write(in_f.read())


def _load_serving_shards_meta(shards_dir: str | Path) -> dict[str, object] | None:
    meta_path = Path(shards_dir) / SERVING_SHARD_META_FILE
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    shard_count = payload.get("shard_count")
    if not isinstance(shard_count, int) or shard_count <= 0:
        return None
    payload.setdefault("format", "csv.gz")
    payload.setdefault("hash_algo", "blake2b64")
    return payload


def _resolve_shard_count(serving_shards_dir: str | Path | None, fallback: int) -> int:
    if not serving_shards_dir:
        return fallback
    meta = _load_serving_shards_meta(serving_shards_dir)
    if meta and isinstance(meta.get("shard_count"), int):
        return int(meta["shard_count"])
    return fallback


def build_serving_shards(
    month_to_file: Mapping[str, str | Path],
    *,
    shards_dir: str | Path,
    shard_count: int = DEFAULT_SERVING_SHARD_COUNT,
    src_col: str = "src",
    dst_col: str = "dst",
    csv_delimiter: str = ",",
    csv_has_header: bool = True,
    latest_only: bool = True,
    show_progress: bool = False,
    force_rebuild: bool = False,
    log_progress: bool = False,
    workers: int = 1,
) -> Path:
    if shard_count <= 0:
        raise ValueError("shard_count must be a positive integer")
    if not csv_has_header:
        raise ValueError("--build-serving-shards requires CSV headers")
    if workers < 1:
        raise ValueError("workers must be >= 1")

    month_paths = normalize_month_paths(month_to_file)
    if not month_paths:
        raise ValueError("No month sources provided. Use --credibench and/or --month-file.")

    month_plan = [_latest_month_for_map(month_paths)] if latest_only else list(MONTHS)
    resolved_shards_dir = Path(shards_dir).resolve()

    if resolved_shards_dir.exists() and force_rebuild:
        shutil.rmtree(resolved_shards_dir)
    resolved_shards_dir.mkdir(parents=True, exist_ok=True)

    lock_fd = _acquire_shard_build_lock(resolved_shards_dir)

    build_tmp = resolved_shards_dir / _SHARD_BUILD_TMP_DIR
    progress_file = build_tmp / _SHARD_BUILD_PROGRESS_FILE

    import signal as _signal
    _old_sigterm = _signal.getsignal(_signal.SIGTERM)

    def _sigterm_handler(signum, frame):
        logger.warning("[shards] SIGTERM received — progress saved up to last completed month")
        _release_shard_build_lock(lock_fd)
        if callable(_old_sigterm):
            _old_sigterm(signum, frame)
        else:
            raise SystemExit(128 + signum)

    _signal.signal(_signal.SIGTERM, _sigterm_handler)

    try:
        done_months: set[str] = set()
        build_tmp.mkdir(parents=True, exist_ok=True)
        if progress_file.exists():
            try:
                progress_data = json.loads(progress_file.read_text(encoding="utf-8"))
                done_months = set(progress_data.get("done_months", []))
                if log_progress and done_months:
                    logger.info(
                        "[shards] resuming — %d month(s) already done: %s",
                        len(done_months),
                        sorted(done_months),
                    )
            except Exception:
                done_months = set()

        for shard_id in range(shard_count):
            shard_dir = resolved_shards_dir / f"shard_id={shard_id}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / "neighbors.csv.gz"
            if not shard_path.exists():
                with gzip.open(shard_path, "wt", encoding="utf-8", newline="") as h:
                    csv.writer(h).writerow(["domain", "month", "neighbor"])

        remaining = [
            m for m in month_plan
            if m not in done_months
            and month_paths.get(m) is not None
            and (not isinstance(month_paths[m], Path) or month_paths[m].exists())
        ]

        start = time.monotonic()
        total_edges = 0

        if remaining:
            month_bar = None
            if show_progress:
                try:
                    from tqdm import tqdm as _tqdm  # type: ignore
                    month_bar = _tqdm(total=len(remaining), desc="serving shard build", unit="month")
                except Exception:
                    pass

            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(
                            _process_month_to_tmp,
                            month,
                            month_paths[month],
                            build_tmp,
                            shard_count,
                            src_col,
                            dst_col,
                            csv_delimiter,
                            csv_has_header,
                            show_progress,
                            log_progress,
                        ): month
                        for month in remaining
                    }

                    for future in as_completed(futures):
                        month = futures[future]
                        try:
                            edge_count = future.result()
                        except Exception as exc:
                            logger.error("[shards] failed to process month=%s: %s", month, exc)
                            if month_bar is not None:
                                month_bar.update(1)
                            continue

                        total_edges += edge_count
                        _merge_month_into_shards(month, build_tmp, resolved_shards_dir, shard_count, log_progress)
                        done_months.add(month)
                        progress_file.write_text(
                            json.dumps({"done_months": sorted(done_months)}, indent=2),
                            encoding="utf-8",
                        )
                        if log_progress:
                            logger.info(
                                "[shards] month=%s merged (%s edges total so far)",
                                month,
                                f"{total_edges:,}",
                            )
                        if month_bar is not None:
                            month_bar.update(1)
            finally:
                if month_bar is not None:
                    month_bar.close()
        elif log_progress:
            logger.info("[shards] all months already done — writing meta and exiting")

        meta = {
            "shard_count": shard_count,
            "created_at_unix": int(time.time()),
            "format": "csv.gz",
            "hash_algo": "blake2b64",
            "fields": ["domain", "month", "neighbor"],
            "source": "month_map_streaming",
            "latest_only": latest_only,
            "months": month_plan,
        }
        (resolved_shards_dir / SERVING_SHARD_META_FILE).write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        shutil.rmtree(build_tmp, ignore_errors=True)

        if log_progress:
            logger.info(
                "[shards] done — %s directed edges, %d shard(s), %.1fs",
                f"{total_edges:,}",
                shard_count,
                time.monotonic() - start,
            )
    finally:
        _release_shard_build_lock(lock_fd)
        try:
            _signal.signal(_signal.SIGTERM, _old_sigterm)
        except Exception:
            pass

    return resolved_shards_dir


def _query_temporal_neighbors(
    domains: list[str],
    *,
    serving_shards_dir: str | Path | None,
    shard_count: int,
    month_map: Mapping[str, str | Path],
    src_col: str,
    dst_col: str,
    csv_delimiter: str,
    csv_has_header: bool,
    months_back: int = 0,
    max_neighbors_per_month: int = 0,
) -> dict[str, dict[str, list[str]]]:
    month_plan = list(MONTHS)
    if months_back > 0:
        month_plan = list(MONTHS[-months_back:])

    if serving_shards_dir:
        return _batch_one_hop_neighbors_by_month_from_serving_shards(
            domains,
            serving_shards_dir=serving_shards_dir,
            shard_count=shard_count,
            months=month_plan,
            max_neighbors_per_month=max_neighbors_per_month,
        )
    return batch_one_hop_neighbors_by_month(
        domains,
        month_map,
        src_col=src_col,
        dst_col=dst_col,
        csv_delimiter=csv_delimiter,
        csv_has_header=csv_has_header,
        months=month_plan,
        max_neighbors_per_month=max_neighbors_per_month,
    )

def _query_latest_neighbors(
    domains: list[str],
    *,
    serving_shards_dir: str | Path | None,
    shard_count: int,
    month_map: Mapping[str, str | Path],
    hops: int,
    max_domains_per_hop: int,
    src_col: str,
    dst_col: str,
    csv_delimiter: str,
    csv_has_header: bool,
) -> dict[str, list[str]]:
    if serving_shards_dir:
        return _latest_two_hop_from_serving_shards(
            domains,
            serving_shards_dir=serving_shards_dir,
            shard_count=shard_count,
            hops=hops,
            max_domains_per_hop=max_domains_per_hop,
        )
    return _latest_two_hop_from_month_map(
        domains,
        month_map,
        hops=hops,
        max_domains_per_hop=max_domains_per_hop,
        src_col=src_col,
        dst_col=dst_col,
        csv_delimiter=csv_delimiter,
        csv_has_header=csv_has_header,
    )


def _batch_one_hop_neighbors_by_month_from_serving_shards(
    domains: list[str],
    *,
    serving_shards_dir: str | Path,
    shard_count: int,
    months: list[str] | None = None,
    max_neighbors_per_month: int = 0,
) -> dict[str, dict[str, list[str]]]:
    input_domains = sorted({d.strip().lower() for d in domains if d.strip()})
    if not input_domains:
        raise ValueError("domains must contain at least one non-empty domain")

    variant_sets: dict[str, set[str]] = {}
    shard_variant_to_inputs: dict[int, dict[str, set[str]]] = {}
    for input_domain in input_domains:
        variants = set(_domain_variants(input_domain))
        variant_sets[input_domain] = variants
        for variant in variants:
            shard_id = _stable_shard_id(variant, shard_count)
            shard_variant_to_inputs.setdefault(shard_id, {}).setdefault(variant, set()).add(input_domain)

    month_plan = months if months else list(MONTHS)
    month_lookup = set(month_plan)
    result_sets: dict[str, dict[str, set[str]]] = {
        domain: {month: set() for month in month_plan}
        for domain in input_domains
    }

    for shard_id, variant_map in shard_variant_to_inputs.items():
        shard_file = Path(serving_shards_dir) / f"shard_id={shard_id}" / "neighbors.csv.gz"
        if not shard_file.exists():
            continue
        with gzip.open(shard_file, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            try:
                for row in reader:
                    domain = str(row.get("domain", "")).strip().lower()
                    month = str(row.get("month", "")).strip().lower()
                    neighbor = str(row.get("neighbor", "")).strip().lower()
                    if domain not in variant_map or month not in month_lookup or not neighbor:
                        continue
                    for input_domain in variant_map[domain]:
                        if neighbor in variant_sets[input_domain]:
                            continue
                        if max_neighbors_per_month > 0 and len(result_sets[input_domain][month]) >= max_neighbors_per_month:
                            continue
                        result_sets[input_domain][month].add(neighbor)
            except EOFError:
                logger.warning("[shards] truncated gzip shard %s — partial data used", shard_file)

    return {
        domain: {month: sorted(result_sets[domain][month]) for month in month_plan}
        for domain in input_domains
    }


def _latest_two_hop_from_month_map(
    domains: list[str],
    month_map: Mapping[str, str | Path],
    *,
    hops: int,
    max_domains_per_hop: int,
    src_col: str,
    dst_col: str,
    csv_delimiter: str,
    csv_has_header: bool,
) -> dict[str, list[str]]:
    monthly = batch_one_hop_neighbors_by_month(
        domains,
        month_map,
        src_col=src_col,
        dst_col=dst_col,
        csv_delimiter=csv_delimiter,
        csv_has_header=csv_has_header,
    )
    latest_month = _latest_month_for_map(month_map)

    first_hop: dict[str, set[str]] = {}
    for domain in domains:
        neighbors = set(sorted(monthly.get(domain, {}).get(latest_month, []))[:max_domains_per_hop])
        neighbors -= set(_domain_variants(domain))
        first_hop[domain] = neighbors

    if hops <= 1:
        return {domain: sorted(first_hop[domain]) for domain in domains}

    expand_domains = sorted({n for vals in first_hop.values() for n in vals})[:max_domains_per_hop]
    if not expand_domains:
        return {domain: sorted(first_hop[domain]) for domain in domains}

    second_monthly = batch_one_hop_neighbors_by_month(
        expand_domains,
        month_map,
        src_col=src_col,
        dst_col=dst_col,
        csv_delimiter=csv_delimiter,
        csv_has_header=csv_has_header,
    )

    result: dict[str, list[str]] = {}
    for domain in domains:
        combined = set(first_hop[domain])
        for neighbor in first_hop[domain]:
            combined.update(sorted(second_monthly.get(neighbor, {}).get(latest_month, []))[:max_domains_per_hop])
        combined -= set(_domain_variants(domain))
        result[domain] = sorted(combined)[: max_domains_per_hop * max(1, hops)]
    return result


def _latest_two_hop_from_serving_shards(
    domains: list[str],
    *,
    serving_shards_dir: str | Path,
    shard_count: int,
    hops: int,
    max_domains_per_hop: int,
) -> dict[str, list[str]]:
    monthly = _batch_one_hop_neighbors_by_month_from_serving_shards(
        domains,
        serving_shards_dir=serving_shards_dir,
        shard_count=shard_count,
    )
    meta = _load_serving_shards_meta(serving_shards_dir)
    shard_months = meta.get("months") if meta else None
    if shard_months and isinstance(shard_months, list):
        latest_month = next(
            (m for m in reversed(MONTHS) if m in shard_months),
            MONTHS[-1],
        )
    else:
        latest_month = MONTHS[-1]

    first_hop: dict[str, set[str]] = {}
    for domain in domains:
        neighbors = set(sorted(monthly.get(domain, {}).get(latest_month, []))[:max_domains_per_hop])
        neighbors -= set(_domain_variants(domain))
        first_hop[domain] = neighbors

    if hops <= 1:
        return {domain: sorted(first_hop[domain]) for domain in domains}

    expand_domains = sorted({n for vals in first_hop.values() for n in vals})[:max_domains_per_hop]
    if not expand_domains:
        return {domain: sorted(first_hop[domain]) for domain in domains}

    second_monthly = _batch_one_hop_neighbors_by_month_from_serving_shards(
        expand_domains,
        serving_shards_dir=serving_shards_dir,
        shard_count=shard_count,
    )

    result: dict[str, list[str]] = {}
    for domain in domains:
        combined = set(first_hop[domain])
        for neighbor in first_hop[domain]:
            combined.update(sorted(second_monthly.get(neighbor, {}).get(latest_month, []))[:max_domains_per_hop])
        combined -= set(_domain_variants(domain))
        result[domain] = sorted(combined)[: max_domains_per_hop * max(1, hops)]
    return result


def _read_domains_file(path: str | Path) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"domains file not found: {file_path}")
    suffixes = [suffix.lower() for suffix in file_path.suffixes]
    if suffixes and suffixes[-1] == ".csv":
        domains: list[str] = []
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for i, row in enumerate(reader):
                if not row:
                    continue
                domain = row[0].strip().lower()
                if not domain:
                    continue
                if i == 0 and domain in {"domain", "domains"}:
                    continue
                domains.append(domain)
        return domains
    return [line.strip().lower() for line in file_path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Return neighborhood lookups from edge files or serving shards. "
            "Default query mode is 'latest' (multi-hop on the most recent month)."
        )
    )
    parser.add_argument("domain", nargs="?", help="Domain to query, e.g. bbc.com")
    parser.add_argument(
        "--month-file",
        action="append",
        default=[],
        metavar="MONTH=PATH",
        help="Provide month-file mapping. Example: --month-file oct2024=/data/oct.csv.gz",
    )
    parser.add_argument(
        "--build-serving-shards",
        help="Build CSV.GZ serving shards from month sources and exit",
    )
    parser.add_argument(
        "--serving-shards-dir",
        help="Directory containing serving shard files",
    )
    parser.add_argument(
        "--serving-shard-count",
        type=int,
        default=DEFAULT_SERVING_SHARD_COUNT,
        help="Number of serving shard partitions to build/use (default: 256)",
    )
    parser.add_argument(
        "--all-months",
        action="store_true",
        help="For --build-serving-shards, include all months (default is latest month only)",
    )
    parser.add_argument(
        "--force-rebuild-serving-shards",
        action="store_true",
        help="Delete and rebuild serving shards",
    )
    parser.add_argument(
        "--shard-workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel month-processing workers for --build-serving-shards (default: 1)",
    )
    parser.add_argument(
        "--domains-file",
        help="One domain per line for batch query output",
    )
    parser.add_argument(
        "--mode",
        choices=["latest", "temporal"],
        default=DEFAULT_QUERY_MODE,
        help=(
            "Query mode. Default is 'latest' (latest month with k-hop expansion); "
            "use 'temporal' for month-by-month one-hop output"
        ),
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=DEFAULT_HOPS,
        help="Neighborhood hops for latest mode (default: 2)",
    )
    parser.add_argument(
        "--max-domains-per-hop",
        type=int,
        default=DEFAULT_MAX_DOMAINS_PER_HOP,
        help="Cap on number of domains expanded per hop in latest mode (default: 10)",
    )
    parser.add_argument(
        "--temporal-months-back",
        type=int,
        default=0,
        help="For temporal mode, only include the latest N months (0 means all months)",
    )
    parser.add_argument(
        "--max-neighbors-per-month",
        type=int,
        default=0,
        help="For temporal mode, cap neighbors kept per month (0 means no cap)",
    )
    parser.add_argument(
        "--credibench",
        action="store_true",
        help="Read monthly edges directly from the CrediBench dataset on Hugging Face",
    )
    parser.add_argument(
        "--hf-repo",
        default=CREDIBENCH_REPO_ID,
        help="Hugging Face dataset repo id to use with --credibench",
    )
    parser.add_argument(
        "--hf-revision",
        default="main",
        help="Hugging Face dataset revision to use with --credibench",
    )
    parser.add_argument("--src-col", default="src", help="Source column name")
    parser.add_argument("--dst-col", default="dst", help="Destination column name")
    parser.add_argument("--csv-delimiter", default=",", help="CSV delimiter")
    parser.add_argument(
        "--csv-no-header",
        action="store_true",
        help="Set when CSV files do not have a header row",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Emit progress logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%H:%M:%S",
        )

    if args.serving_shard_count <= 0:
        parser.error("--serving-shard-count must be a positive integer")
    if args.hops <= 0:
        parser.error("--hops must be a positive integer")
    if args.max_domains_per_hop <= 0:
        parser.error("--max-domains-per-hop must be a positive integer")
    if args.temporal_months_back < 0:
        parser.error("--temporal-months-back must be >= 0")
    if args.max_neighbors_per_month < 0:
        parser.error("--max-neighbors-per-month must be >= 0")

    month_map = parse_month_file_args(args.month_file)
    if args.credibench:
        month_map = {
            **credibench_month_urls(repo_id=args.hf_repo, revision=args.hf_revision),
            **month_map,
        }

    if args.build_serving_shards:
        if not month_map:
            parser.error("--build-serving-shards requires month sources via --credibench and/or --month-file")
        out_dir = build_serving_shards(
            month_map,
            shards_dir=args.build_serving_shards,
            shard_count=args.serving_shard_count,
            src_col=args.src_col,
            dst_col=args.dst_col,
            csv_delimiter=args.csv_delimiter,
            csv_has_header=not args.csv_no_header,
            latest_only=not args.all_months,
            show_progress=args.verbose,
            force_rebuild=args.force_rebuild_serving_shards,
            log_progress=args.verbose,
            workers=args.shard_workers,
        )
        print(f"Built serving shards at {out_dir}")
        return

    if args.domains_file:
        domains = _read_domains_file(args.domains_file)
        shard_count = _resolve_shard_count(args.serving_shards_dir, args.serving_shard_count)
        if args.mode == "temporal":
            result = _query_temporal_neighbors(
                domains,
                serving_shards_dir=args.serving_shards_dir,
                shard_count=shard_count,
                month_map=month_map,
                src_col=args.src_col,
                dst_col=args.dst_col,
                csv_delimiter=args.csv_delimiter,
                csv_has_header=not args.csv_no_header,
                months_back=args.temporal_months_back,
                max_neighbors_per_month=args.max_neighbors_per_month,
            )
        else:
            result = _query_latest_neighbors(
                domains,
                serving_shards_dir=args.serving_shards_dir,
                shard_count=shard_count,
                month_map=month_map,
                hops=args.hops,
                max_domains_per_hop=args.max_domains_per_hop,
                src_col=args.src_col,
                dst_col=args.dst_col,
                csv_delimiter=args.csv_delimiter,
                csv_has_header=not args.csv_no_header,
            )
        if args.pretty:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        return

    if not args.domain:
        parser.error("domain is required unless using --build-serving-shards or --domains-file")

    query_domain = args.domain.strip().lower()
    shard_count = _resolve_shard_count(args.serving_shards_dir, args.serving_shard_count)

    if args.mode == "temporal":
        result = _query_temporal_neighbors(
            [query_domain],
            serving_shards_dir=args.serving_shards_dir,
            shard_count=shard_count,
            month_map=month_map,
            src_col=args.src_col,
            dst_col=args.dst_col,
            csv_delimiter=args.csv_delimiter,
            csv_has_header=not args.csv_no_header,
            months_back=args.temporal_months_back,
            max_neighbors_per_month=args.max_neighbors_per_month,
        )[query_domain]
    else:
        result = _query_latest_neighbors(
            [query_domain],
            serving_shards_dir=args.serving_shards_dir,
            shard_count=shard_count,
            month_map=month_map,
            hops=args.hops,
            max_domains_per_hop=args.max_domains_per_hop,
            src_col=args.src_col,
            dst_col=args.dst_col,
            csv_delimiter=args.csv_delimiter,
            csv_has_header=not args.csv_no_header,
        )[query_domain]

    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(json.dumps(result, separators=(",", ":"), sort_keys=False))


if __name__ == "__main__":
    main()
