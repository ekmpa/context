import argparse
import csv
import os
from pathlib import Path

import hook
from context_core.utils import search

try:
    from tqdm.auto import tqdm
except Exception: 
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prewarm the structural neighborhood cache.")
    parser.add_argument(
        "--domains-file",
        default="",
        help="Optional file with domains or URLs. Supports one-per-line text or CSV.",
    )
    parser.add_argument(
        "--domain-ratings-file",
        default="data/domain_ratings.csv",
        help="Fallback CSV to read domains from when --domains-file is not provided.",
    )
    parser.add_argument(
        "--domain-column",
        default="domain",
        help="CSV column name to read domains or URLs from.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="How many domains to prefetch in one batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of domains to prewarm.",
    )
    parser.add_argument(
        "--seed-batch-size",
        type=int,
        default=64,
        help="How many seed domains to expand at once when collecting 1-hop neighbors.",
    )
    parser.add_argument(
        "--skip-one-hop",
        action="store_true",
        help="Only prewarm the seed domains; do not expand their 1-hop temporal neighbors.",
    )
    return parser.parse_args()


def _iter_domains_from_text(path: Path) -> list[str]:
    domains: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        domain = search._extract_domain(line.strip())
        if domain:
            domains.append(domain)
    return domains


def _iter_domains_from_csv(path: Path, column: str) -> list[str]:
    domains: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            domain = search._extract_domain(str(row.get(column, "") or ""))
            if domain:
                domains.append(domain)
    return domains


def _load_domains(args: argparse.Namespace) -> list[str]:
    input_path = Path(args.domains_file or args.domain_ratings_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Domain source not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        domains = _iter_domains_from_csv(input_path, args.domain_column)
    else:
        domains = _iter_domains_from_text(input_path)

    unique_domains = list(dict.fromkeys(domains))
    if args.limit and args.limit > 0:
        unique_domains = unique_domains[: args.limit]
    return unique_domains


def _progress(iterable, *, total: int | None = None, desc: str = ""):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _serving_shards_dir() -> str | None:
    shards_dir = os.getenv("RARR_STRUCTURAL_SHARDS_DIR", "").strip()
    return shards_dir or None


def _expand_one_hop_neighbors(seed_domains: list[str], seed_batch_size: int) -> list[str]:
    if not seed_domains:
        return []

    shard_count = hook._resolve_shard_count(_serving_shards_dir(), hook.DEFAULT_SERVING_SHARD_COUNT)
    month_map = hook.parse_month_file_args([])
    months_back = max(0, int(os.getenv("RARR_STRUCTURAL_MONTHS_BACK", "3") or "3"))
    max_neighbors = max(0, int(os.getenv("RARR_STRUCTURAL_MAX_NEIGHBORS", "8") or "8"))

    discovered: list[str] = []
    seen = set(seed_domains)
    total_batches = (len(seed_domains) + seed_batch_size - 1) // seed_batch_size
    batch_starts = range(0, len(seed_domains), seed_batch_size)
    for start in _progress(batch_starts, total=total_batches, desc="expand 1-hop"):
        chunk = seed_domains[start:start + seed_batch_size]
        monthly = hook._query_temporal_neighbors(
            chunk,
            serving_shards_dir=_serving_shards_dir(),
            shard_count=shard_count,
            month_map=month_map,
            src_col="src",
            dst_col="dst",
            csv_delimiter=",",
            csv_has_header=True,
            months_back=months_back,
            max_neighbors_per_month=max_neighbors,
        )
        for domain in chunk:
            for month, neighbors in (monthly.get(domain) or {}).items():
                for neighbor in neighbors:
                    normalized = search._extract_domain(neighbor)
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    discovered.append(normalized)
    return discovered


def main() -> None:
    args = parse_args()
    seed_domains = _load_domains(args)
    if not seed_domains:
        print("No domains found to prewarm.")
        return

    one_hop_domains: list[str] = []
    if not args.skip_one_hop:
        one_hop_domains = _expand_one_hop_neighbors(
            seed_domains,
            max(1, int(args.seed_batch_size)),
        )

    domains = [*seed_domains, *one_hop_domains]
    total = len(domains)

    chunk_size = max(1, int(args.chunk_size))
    print(
        f"Prewarming structural cache for {total} domains "
        f"(seeds={len(seed_domains)}, one_hop={len(one_hop_domains)}, chunk_size={chunk_size})"
    )
    batch_starts = range(0, total, chunk_size)
    total_batches = (total + chunk_size - 1) // chunk_size
    for start in _progress(batch_starts, total=total_batches, desc="prefetch cache"):
        chunk = domains[start:start + chunk_size]
        contexts = search._prefetch_structural_contexts(chunk)
        nonempty = sum(1 for value in contexts.values() if (value or "").strip())
        if tqdm is None:
            print(
                f"chunk {start // chunk_size + 1}: domains={len(chunk)} "
                f"cached_with_context={nonempty} total_done={min(start + len(chunk), total)}/{total}"
            )

    search._persist_structural_cache_file(force=True)
    print(f"Done. Cache file: {search._get_structural_cache_file()}")


if __name__ == "__main__":
    main()