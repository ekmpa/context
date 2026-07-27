"""Utils for searching a query and returning top passages from search results."""
import atexit
import concurrent.futures
import csv
import fcntl
import itertools
import json
import logging
import os
import random
import subprocess
import sys
import time
import threading
from collections import Counter
from typing import Any, Dict, List, Tuple
from pathlib import Path
from urllib.parse import urlparse

import bs4
import requests
import spacy
import torch
from sentence_transformers import CrossEncoder

PASSAGE_RANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
    device="cpu",
)
SERPER_SEARCH_URL = "https://google.serper.dev/search"
TOKENIZER = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
STRUCTURAL_MONTH_ORDER = (
    "apr2025",
    "mar2025",
    "feb2025",
    "jan2025",
    "dec2024",
    "nov2024",
    "oct2024",
)
STRUCTURAL_CACHE: Dict[str, str] = {}
STRUCTURAL_CACHE_LOADED = False
STRUCTURAL_CACHE_DIRTY = False
STRUCTURAL_CACHE_PENDING_WRITES = 0
STRUCTURAL_CACHE_UPDATES_OFFSET = 0
STRUCTURAL_CACHE_LOCK = threading.Lock()
SEARCH_STATS: Dict[str, int] = {
    "search_timeouts": 0,
    "serper_timeout_fallback_to_ddg": 0,
    "serper_empty_fallback_to_ddg": 0,
    "provider_failures": 0,
    "queries_with_no_results": 0,
    "queries_discarded": 0,
    "structural_queries_total": 0,
    "structural_queries_with_context": 0,
    "structural_evidence_total": 0,
    "structural_evidence_with_context": 0,
    "structural_cache_hits": 0,
    "structural_cache_misses": 0,
    "structural_hook_failures": 0,
    "structural_hop_fallbacks": 0,
}
THIRD_PARTY_RATINGS: Dict[str, Dict[str, float]] = {}
THIRD_PARTY_RATINGS_LOADED = False
logger = logging.getLogger(__name__)


def _record_search_stat(key: str, value: int = 1) -> None:
    SEARCH_STATS[key] = SEARCH_STATS.get(key, 0) + value


class SearchQueryFailed(RuntimeError):
    """Raised when a query still fails after provider retries."""


def _search_retry_wait_seconds(attempt: int, base_wait: float = 1.0) -> float:
    return base_wait * (2 ** max(0, attempt - 1))


def reset_search_stats() -> None:
    for key in list(SEARCH_STATS.keys()):
        SEARCH_STATS[key] = 0


def get_search_stats() -> Dict[str, int]:
    return dict(SEARCH_STATS)


def _is_timeout_exception(exc: Exception) -> bool:
    return isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ReadTimeout))


def _condition_mode() -> str:
    mode = os.getenv("RARR_CONDITION", "").strip().lower()
    if mode in {"raw", "structural", "third-party", "source_attr"}:
        return mode
    raise RuntimeError(
        "RARR_CONDITION must be one of: raw, structural, third-party, source_attr"
    )


def _is_missing_text(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "na", "none", "null", "nan", "n/a"}


def _registrable_domain(host: str) -> str:
    host = (host or "").strip().lower().strip(".")
    if not host:
        return ""
    if host.startswith("www."):
        host = host[4:]

    parts = [p for p in host.split(".") if p]
    if len(parts) <= 2:
        return host

    second_level_cc = {"co", "com", "org", "net", "gov", "edu", "ac"}
    cc_tlds = {"uk", "au", "jp", "nz", "za"}
    if len(parts) >= 3 and parts[-1] in cc_tlds and parts[-2] in second_level_cc:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").split(":")[0].strip().lower()
    if not host:
        # Allow plain domains without scheme, e.g. "reuters.com".
        host = (parsed.path or "").split("/")[0].split(":")[0].strip().lower()
    return _registrable_domain(host)


def _third_party_ratings_path() -> str:
    configured = os.getenv("RARR_THIRD_PARTY_RATINGS_FILE", "").strip()
    if configured:
        return configured
    return str(Path(__file__).resolve().parents[3] / "data" / "domain_ratings.csv")


def _safe_float(value: Any) -> float | None:
    try:
        if _is_missing_text(value):
            return None
        return float(str(value).strip())
    except Exception:
        return None


def _load_third_party_ratings() -> None:
    global THIRD_PARTY_RATINGS_LOADED
    if THIRD_PARTY_RATINGS_LOADED:
        return
    THIRD_PARTY_RATINGS_LOADED = True

    ratings_file = _third_party_ratings_path()
    if not os.path.isfile(ratings_file):
        logger.warning("[search] third-party ratings file not found: %s", ratings_file)
        return

    try:
        with open(ratings_file, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                domain = _extract_domain(str(row.get("domain", "")))
                if not domain:
                    continue
                pc1 = _safe_float(row.get("pc1"))
                afm_bias = _safe_float(row.get("afm_bias"))
                mbfc_bias = _safe_float(row.get("mbfc_bias"))
                if afm_bias is not None and mbfc_bias is not None:
                    bias_score = (afm_bias + mbfc_bias) / 2.0
                elif afm_bias is not None:
                    bias_score = afm_bias
                elif mbfc_bias is not None:
                    bias_score = mbfc_bias
                else:
                    bias_score = None

                THIRD_PARTY_RATINGS[domain] = {
                    "pc1": pc1,
                    "bias_score": bias_score,
                }
    except Exception as exc:
        logger.warning("[search] failed to load third-party ratings file %s: %s", ratings_file, exc)


def _format_source_attr_context(domain: str) -> str:
    """Return a source-attribution prompt fragment for the given domain."""
    if not domain:
        return ""
    return (
        f"Source attribution checklist for '{domain}':\n"
        "- Publisher/Context: who is behind this domain and what is the publication context?\n"
        "- Source Type: is this a primary source or a secondary source for this claim?\n"
        "- Conflict of Interest: any plausible stake or incentive that could bias framing?\n"
        "- Outlet Category: news, government, academic, advocacy, commercial, or social platform?"
    )


def _lookup_third_party_context(domain: str) -> str:
    if not domain:
        return ""
    _load_third_party_ratings()

    scores = THIRD_PARTY_RATINGS.get(domain)
    if not scores:
        return ""

    reliability_score = scores.get("pc1")
    if reliability_score is None:
        # Missing scores should behave like raw mode (no extra context appended).
        return ""

    return (
        f"Source note: '{domain}' has a trust score of {reliability_score:.4f} "
        f"(scores span [0,1], higher is better)."
    )


def _format_structural_context(domain: str, payload: Any) -> str:
    max_neighbors = int(os.getenv("RARR_STRUCTURAL_MAX_NEIGHBORS", "8"))
    months_back = max(1, int(os.getenv("RARR_STRUCTURAL_MONTHS_BACK", "3")))

    if isinstance(payload, list):
        neighbors = [x for x in payload if isinstance(x, str)][:max_neighbors]
        if not neighbors:
            return f"Source context for {domain}: no co-linking neighbors found in graph index."
        return (
            f"Source context for {domain}: mode=latest, peers={len(neighbors)}, "
            f"top_peers={', '.join(neighbors)}. Treat as a weak prior unless corroborated by evidence text."
        )

    if not isinstance(payload, dict):
        return f"Source context for {domain}: graph data format not recognized."

    available_months = [
        month for month in STRUCTURAL_MONTH_ORDER
        if isinstance(payload.get(month), list) and payload.get(month)
    ]
    selected_months = available_months[:months_back]
    if not selected_months:
        return f"Source context for {domain}: no neighbor data found in graph index."

    month_sizes = {month: len(payload.get(month, [])) for month in selected_months}
    peer_counter: Counter[str] = Counter()
    for month in selected_months:
        month_neighbors = [x for x in payload.get(month, []) if isinstance(x, str)]
        peer_counter.update(month_neighbors)

    unique_peers = len(peer_counter)
    peer_mentions = sum(peer_counter.values())
    repeated_peers = sum(1 for _, count in peer_counter.items() if count >= 2)
    top_peers = sorted(peer_counter.items(), key=lambda item: (-item[1], item[0]))[:max_neighbors]
    top_peer_text = ", ".join(f"{peer}({count})" for peer, count in top_peers) if top_peers else "none"
    month_size_text = ", ".join(f"{month}:{month_sizes[month]}" for month in selected_months)

    return (
        f"Source context for {domain}: mode=temporal, months_considered={len(selected_months)}, "
        f"month_neighbor_counts=[{month_size_text}], unique_peers={unique_peers}, "
        f"peer_mentions={peer_mentions}, repeated_peers={repeated_peers}, "
        f"top_peers={top_peer_text}. Higher repeated_peers and cross-month consistency indicate stronger structural prior."
    )


def _lookup_structural_context(domain: str, timeout: float | None = None) -> str:
    global STRUCTURAL_CACHE_DIRTY, STRUCTURAL_CACHE_PENDING_WRITES
    if not domain:
        return ""

    hook_mode = os.getenv("RARR_STRUCTURAL_HOOK_MODE", "temporal").strip().lower()
    if hook_mode not in {"latest", "temporal"}:
        hook_mode = "temporal"

    configured_hops_raw = os.getenv("RARR_STRUCTURAL_HOPS", "2").strip()
    try:
        configured_hops = max(1, int(configured_hops_raw))
    except Exception:
        configured_hops = 2

    def _cache_key_for(domain_name: str) -> str:
        return "|".join(
            [
                domain_name,
                f"mode={hook_mode}",
                f"hops={configured_hops}",
                f"max_domains_per_hop={os.getenv('RARR_STRUCTURAL_MAX_DOMAINS_PER_HOP', '10').strip()}",
                f"months_back={os.getenv('RARR_STRUCTURAL_MONTHS_BACK', '3').strip()}",
                f"max_neighbors={os.getenv('RARR_STRUCTURAL_MAX_NEIGHBORS', '8').strip()}",
            ]
        )

    def _payload_has_neighbors(payload: Any) -> bool:
        if isinstance(payload, list):
            return any(isinstance(item, str) and item.strip() for item in payload)
        if isinstance(payload, dict):
            for neighbors in payload.values():
                if isinstance(neighbors, list) and any(
                    isinstance(item, str) and item.strip() for item in neighbors
                ):
                    return True
        return False

    def _parse_hook_json(raw: str) -> Any:
        try:
            return json.loads(raw)
        except Exception:
            start_candidates = [idx for idx in (raw.find("{"), raw.find("[")) if idx >= 0]
            if not start_candidates:
                raise
            start = min(start_candidates)
            end = max(raw.rfind("}"), raw.rfind("]"))
            if end < start:
                raise
            return json.loads(raw[start : end + 1])

    def _safe_int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)).strip())
        except Exception:
            return default

    def _min_hook_timeout_seconds(mode: str, hops: int) -> float:
        # Graph hook calls over serving shards can take much longer than web search,
        # especially for temporal multi-hop lookups. Apply a conservative floor.
        configured_floor = os.getenv("RARR_STRUCTURAL_HOOK_MIN_TIMEOUT", "").strip()
        if configured_floor:
            try:
                return max(1.0, float(configured_floor))
            except Exception:
                pass

        months_back = max(1, _safe_int_env("RARR_STRUCTURAL_MONTHS_BACK", 3))
        max_domains = max(1, _safe_int_env("RARR_STRUCTURAL_MAX_DOMAINS_PER_HOP", 10))
        max_neighbors = max(1, _safe_int_env("RARR_STRUCTURAL_MAX_NEIGHBORS", 8))

        if mode == "temporal":
            floor = 60.0
            if hops > 1:
                floor = 120.0
            if months_back >= 4:
                floor += 20.0
            if max_domains >= 20:
                floor += 20.0
            if max_neighbors >= 16:
                floor += 20.0
            return floor

        floor = 25.0
        if hops > 1:
            floor = 45.0
        if max_domains >= 20:
            floor += 10.0
        return floor

    hook_path = os.getenv("RARR_STRUCTURAL_HOOK_PATH", "").strip()
    if not hook_path:
        hook_path = str(Path(__file__).resolve().parents[2] / "hook.py")
    if not os.path.isfile(hook_path):
        return ""

    _load_structural_cache_file()
    cache_key = _cache_key_for(domain)
    _refresh_structural_cache_updates()
    with STRUCTURAL_CACHE_LOCK:
        if cache_key in STRUCTURAL_CACHE:
            _record_search_stat("structural_cache_hits")
            return STRUCTURAL_CACHE[cache_key]
        if domain in STRUCTURAL_CACHE:
            _record_search_stat("structural_cache_hits")
            cached = STRUCTURAL_CACHE[domain]
            STRUCTURAL_CACHE[cache_key] = cached
            STRUCTURAL_CACHE_DIRTY = True
            STRUCTURAL_CACHE_PENDING_WRITES += 1
            return cached
    _record_search_stat("structural_cache_misses")

    hook_timeout = timeout
    if hook_timeout is None:
        try:
            hook_timeout = float(os.getenv("RARR_STRUCTURAL_HOOK_TIMEOUT", "20"))
        except Exception:
            hook_timeout = 20.0
    hook_timeout = max(hook_timeout, _min_hook_timeout_seconds(hook_mode, configured_hops))

    def _run_hook(mode: str, timeout_s: float, hops: int) -> Any:
        temporal_fallback_raw = os.getenv(
            "RARR_STRUCTURAL_TEMPORAL_FALLBACK_TO_1HOP",
            "1",
        ).strip().lower()
        temporal_fallback_enabled = temporal_fallback_raw not in {"0", "false", "no", "off"}

        cmd = [
            sys.executable,
            hook_path,
            domain,
            "--mode",
            mode,
            "--hops",
            str(hops),
            "--max-domains-per-hop",
            os.getenv("RARR_STRUCTURAL_MAX_DOMAINS_PER_HOP", "10"),
        ]
        if mode == "temporal":
            cmd.extend(
                [
                    "--temporal-months-back",
                    os.getenv("RARR_STRUCTURAL_MONTHS_BACK", "3"),
                    "--max-neighbors-per-month",
                    os.getenv("RARR_STRUCTURAL_MAX_NEIGHBORS", "8"),
                ]
            )
            if temporal_fallback_enabled:
                cmd.append("--temporal-fallback-to-1hop")
            else:
                cmd.append("--no-temporal-fallback-to-1hop")

        shards_dir = os.getenv("RARR_STRUCTURAL_SHARDS_DIR", "").strip()
        if shards_dir:
            cmd.extend(["--serving-shards-dir", shards_dir])

        raw = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_s,
        )
        return _parse_hook_json(raw)

    payload = None
    used_hop_fallback = False
    try:
        payload = _run_hook(hook_mode, hook_timeout, configured_hops)
    except Exception:
        _record_search_stat("structural_hook_failures")

    if configured_hops > 1 and (payload is None or not _payload_has_neighbors(payload)):
        try:
            payload = _run_hook(hook_mode, hook_timeout, 1)
            used_hop_fallback = True
            _record_search_stat("structural_hop_fallbacks")
        except Exception:
            _record_search_stat("structural_hook_failures")

    if payload is None and hook_mode == "latest":
        fallback_timeout = hook_timeout
        try:
            fallback_timeout = float(os.getenv("RARR_STRUCTURAL_HOOK_FALLBACK_TIMEOUT", "30"))
        except Exception:
            fallback_timeout = hook_timeout
        fallback_timeout = max(fallback_timeout, _min_hook_timeout_seconds("temporal", configured_hops))

        try:
            payload = _run_hook("temporal", fallback_timeout, configured_hops)
        except Exception:
            _record_search_stat("structural_hook_failures")

        if configured_hops > 1 and (payload is None or not _payload_has_neighbors(payload)):
            try:
                payload = _run_hook("temporal", fallback_timeout, 1)
                used_hop_fallback = True
                _record_search_stat("structural_hop_fallbacks")
            except Exception:
                _record_search_stat("structural_hook_failures")

    if payload is None:
        return ""

    summary = _format_structural_context(domain, payload)
    if used_hop_fallback:
        summary = f"{summary} fallback_used=1hop."
    with STRUCTURAL_CACHE_LOCK:
        STRUCTURAL_CACHE[cache_key] = summary
        STRUCTURAL_CACHE_DIRTY = True
        STRUCTURAL_CACHE_PENDING_WRITES += 1
    _append_structural_cache_update(cache_key, summary)
    return summary


def _get_structural_cache_file() -> str:
    configured = os.getenv("RARR_STRUCTURAL_CACHE_FILE", "").strip()
    if configured:
        return configured
    return str(Path(__file__).resolve().parents[3] / "data" / "structural_neighbors_cache.json")


def _get_structural_cache_updates_file() -> str:
    cache_file = _get_structural_cache_file()
    return f"{cache_file}.updates.jsonl" if cache_file else ""


def _get_structural_cache_lock_file() -> str:
    cache_file = _get_structural_cache_file()
    return f"{cache_file}.lock" if cache_file else ""


def _acquire_structural_cache_file_lock():
    lock_file = _get_structural_cache_lock_file()
    if not lock_file:
        return None
    os.makedirs(os.path.dirname(lock_file), exist_ok=True)
    handle = open(lock_file, "a+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _release_structural_cache_file_lock(handle) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _load_structural_cache_file() -> None:
    global STRUCTURAL_CACHE_LOADED, STRUCTURAL_CACHE_UPDATES_OFFSET
    if STRUCTURAL_CACHE_LOADED:
        return
    STRUCTURAL_CACHE_LOADED = True

    cache_file = _get_structural_cache_file()
    if not cache_file or not os.path.isfile(cache_file):
        return

    try:
        with open(cache_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(key, str) and isinstance(value, str):
                    STRUCTURAL_CACHE[key] = value
    except Exception:
        return

    _refresh_structural_cache_updates(force=True)
    updates_file = _get_structural_cache_updates_file()
    if updates_file and os.path.isfile(updates_file):
        try:
            STRUCTURAL_CACHE_UPDATES_OFFSET = os.path.getsize(updates_file)
        except Exception:
            STRUCTURAL_CACHE_UPDATES_OFFSET = 0


def _refresh_structural_cache_updates(force: bool = False) -> None:
    global STRUCTURAL_CACHE_UPDATES_OFFSET
    updates_file = _get_structural_cache_updates_file()
    if not updates_file or not os.path.isfile(updates_file):
        return

    try:
        file_size = os.path.getsize(updates_file)
    except Exception:
        return

    start_offset = STRUCTURAL_CACHE_UPDATES_OFFSET
    if force or file_size < start_offset:
        start_offset = 0
    if not force and file_size == start_offset:
        return

    loaded_entries: list[tuple[str, str]] = []
    try:
        with open(updates_file, "r", encoding="utf-8") as handle:
            handle.seek(start_offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                key = payload.get("key")
                value = payload.get("value")
                if isinstance(key, str) and isinstance(value, str):
                    loaded_entries.append((key, value))
            end_offset = handle.tell()
    except Exception:
        return

    with STRUCTURAL_CACHE_LOCK:
        for key, value in loaded_entries:
            STRUCTURAL_CACHE[key] = value
        STRUCTURAL_CACHE_UPDATES_OFFSET = end_offset


def _append_structural_cache_update(cache_key: str, summary: str) -> None:
    updates_file = _get_structural_cache_updates_file()
    if not updates_file:
        return

    lock_handle = _acquire_structural_cache_file_lock()
    try:
        os.makedirs(os.path.dirname(updates_file), exist_ok=True)
        with open(updates_file, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"key": cache_key, "value": summary}, ensure_ascii=True, sort_keys=True)
            )
            handle.write("\n")
    except Exception:
        return
    finally:
        _release_structural_cache_file_lock(lock_handle)


def _persist_structural_cache_file(force: bool = False) -> None:
    global STRUCTURAL_CACHE_DIRTY, STRUCTURAL_CACHE_PENDING_WRITES, STRUCTURAL_CACHE_UPDATES_OFFSET
    cache_file = _get_structural_cache_file()
    updates_file = _get_structural_cache_updates_file()
    if not cache_file:
        return

    if not STRUCTURAL_CACHE_DIRTY:
        return

    autosave_enabled = os.getenv("RARR_STRUCTURAL_CACHE_AUTOSAVE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    flush_every = max(1, int(os.getenv("RARR_STRUCTURAL_CACHE_FLUSH_EVERY", "25")))
    if not force:
        if not autosave_enabled:
            return
        if STRUCTURAL_CACHE_PENDING_WRITES < flush_every:
            return

    try:
        lock_handle = _acquire_structural_cache_file_lock()
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        payload: Dict[str, str] = {}
        if os.path.isfile(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as handle:
                    existing_payload = json.load(handle)
                if isinstance(existing_payload, dict):
                    payload.update(
                        {
                            str(key): str(value)
                            for key, value in existing_payload.items()
                            if isinstance(key, str) and isinstance(value, str)
                        }
                    )
            except Exception:
                pass
        if updates_file and os.path.isfile(updates_file):
            try:
                with open(updates_file, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            update_payload = json.loads(line)
                        except Exception:
                            continue
                        key = update_payload.get("key")
                        value = update_payload.get("value")
                        if isinstance(key, str) and isinstance(value, str):
                            payload[key] = value
            except Exception:
                pass
        with STRUCTURAL_CACHE_LOCK:
            payload.update(dict(STRUCTURAL_CACHE))
        tmp_cache_file = f"{cache_file}.tmp"
        with open(tmp_cache_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(tmp_cache_file, cache_file)
        if updates_file and os.path.exists(updates_file):
            with open(updates_file, "w", encoding="utf-8"):
                pass
        with STRUCTURAL_CACHE_LOCK:
            STRUCTURAL_CACHE.clear()
            STRUCTURAL_CACHE.update(payload)
        STRUCTURAL_CACHE_DIRTY = False
        STRUCTURAL_CACHE_PENDING_WRITES = 0
        STRUCTURAL_CACHE_UPDATES_OFFSET = 0
    except Exception:
        STRUCTURAL_CACHE_DIRTY = True
    finally:
        try:
            _release_structural_cache_file_lock(lock_handle)
        except Exception:
            pass


def _flush_structural_cache_on_exit() -> None:
    try:
        _persist_structural_cache_file(force=True)
    except Exception:
        return


atexit.register(_flush_structural_cache_on_exit)


def _prefetch_structural_contexts(domains: List[str]) -> Dict[str, str]:
    unique_domains: List[str] = []
    seen: set[str] = set()
    for domain in domains:
        if not domain or domain in seen:
            continue
        seen.add(domain)
        unique_domains.append(domain)

    if not unique_domains:
        return {}

    raw_workers = os.getenv("RARR_STRUCTURAL_LOOKUP_WORKERS", "4").strip()
    try:
        workers = max(1, min(int(raw_workers), len(unique_domains)))
    except Exception:
        workers = min(4, len(unique_domains))

    if workers <= 1 or len(unique_domains) <= 1:
        contexts = {domain: _lookup_structural_context(domain) for domain in unique_domains}
        return contexts

    contexts: Dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_domain = {
            pool.submit(_lookup_structural_context, domain): domain for domain in unique_domains
        }
        for future in concurrent.futures.as_completed(future_to_domain):
            domain = future_to_domain[future]
            try:
                contexts[domain] = future.result()
            except Exception:
                contexts[domain] = ""

    return contexts


def chunk_text(
    text: str,
    sentences_per_passage: int,
    filter_sentence_len: int,
    sliding_distance: int = None,
) -> List[str]:
    """Chunks text into passages using a sliding window.

    Args:
        text: Text to chunk into passages.
        sentences_per_passage: Number of sentences for each passage.
        filter_sentence_len: Maximum number of chars of each sentence before being filtered.
        sliding_distance: Sliding distance over the text. Allows the passages to have
            overlap. The sliding distance cannot be greater than the window size.
    Returns:
        passages: Chunked passages from the text.
    """
    if not sliding_distance or sliding_distance > sentences_per_passage:
        sliding_distance = sentences_per_passage
    assert sentences_per_passage > 0 and sliding_distance > 0

    passages = []
    try:
        doc = TOKENIZER(text[:500000])  # Take 500k chars to not break tokenization.
        sents = [
            s.text
            for s in doc.sents
            if len(s.text) <= filter_sentence_len  # Long sents are usually metadata.
        ]
        for idx in range(0, len(sents), sliding_distance):
            passages.append(" ".join(sents[idx : idx + sentences_per_passage]))
    except UnicodeEncodeError as _:  # Sometimes run into Unicode error when tokenizing.
        print("spaCy could not read this text cleanly, so it will be skipped.")

    return passages


def is_tag_visible(element: bs4.element) -> bool:
    """Determines if an HTML element is visible.

    Args:
        element: A BeautifulSoup element to check the visiblity of.
    returns:
        Whether the element is visible.
    """
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ] or isinstance(element, bs4.element.Comment):
        return False
    return True


def scrape_url(url: str, timeout: float = 3) -> Tuple[str, str]:
    """Scrapes a URL for all text information.

    Args:
        url: URL of webpage to scrape.
        timeout: Timeout of the requests call.
    Returns:
        web_text: The visible text of the scraped URL.
        url: URL input.
    """
    # Scrape the URL
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as _:
        return None, url

    # Extract out all text from the tags
    try:
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        texts = soup.findAll(text=True)
        # Filter out invisible text from the page.
        visible_text = filter(is_tag_visible, texts)
    except Exception as _:
        return None, url

    # Returns all the text concatenated as a string.
    web_text = " ".join(t.strip() for t in visible_text).strip()
    # Clean up spacing.
    web_text = " ".join(web_text.split())
    return web_text, url


def search_serper(query: str, timeout: float = 3) -> List[str]:
    """Searches the query using Serper (Google Search API wrapper)."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        raise ValueError("Set SERPER_API_KEY before using the serper search provider.")

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": 10}
    response = requests.post(
        SERPER_SEARCH_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    organic = data.get("organic", [])
    return [item.get("link") for item in organic if item.get("link")]


def search_duckduckgo(query: str, timeout: float = 3) -> List[str]:
    """Searches the query using DuckDuckGo HTML results page (no API key)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": query},
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()

    soup = bs4.BeautifulSoup(response.text, "html.parser")
    results = []
    for link in soup.select("a.result__a"):
        href = link.get("href")
        if href and href.startswith("http"):
            results.append(href)
    return results


def search_web(query: str, timeout: float = 3) -> List[str]:
    """Dispatch web search by provider.

    Providers:
    - serper: requires SERPER_API_KEY
    - duckduckgo: no key
    - auto: serper when configured, otherwise DuckDuckGo
    """
    provider = os.getenv("RARR_SEARCH_PROVIDER", "auto").strip().lower()
    serper_key = os.getenv("SERPER_API_KEY")

    if provider in {"auto", "serper"} and not serper_key:
        if provider == "serper":
            logger.warning("[search] SERPER_API_KEY is missing; using DuckDuckGo instead.")
        return search_duckduckgo(query, timeout=timeout)

    if provider == "serper":
        try:
            results = search_serper(query, timeout=timeout)
            if results:
                return results
            _record_search_stat("serper_empty_fallback_to_ddg")
            logger.warning("[search] Serper returned no results for query %r; trying DuckDuckGo instead.", query)
        except Exception as exc:
            if _is_timeout_exception(exc):
                _record_search_stat("search_timeouts")
                _record_search_stat("serper_timeout_fallback_to_ddg")
                logger.warning("[search] Serper timed out for query %r; trying DuckDuckGo instead.", query)
            else:
                _record_search_stat("provider_failures")
                logger.warning("[search] Serper failed for query %r: %s. Trying DuckDuckGo instead.", query, exc)

        try:
            return search_duckduckgo(query, timeout=timeout)
        except Exception as ddg_exc:
            if _is_timeout_exception(ddg_exc):
                _record_search_stat("search_timeouts")
            _record_search_stat("provider_failures")
            logger.warning(
                "[search] DuckDuckGo failed after Serper attempt for query %r: %s",
                query,
                ddg_exc,
            )
            raise RuntimeError(f"search providers failed for query {query!r}") from ddg_exc
    if provider == "duckduckgo":
        try:
            return search_duckduckgo(query, timeout=timeout)
        except Exception as exc:
            if _is_timeout_exception(exc):
                _record_search_stat("search_timeouts")
                logger.warning("[search] DuckDuckGo timed out for query %r.", query)
                return []
            _record_search_stat("provider_failures")
            raise

    errors = []
    for fn in (search_serper, search_duckduckgo):
        try:
            results = fn(query, timeout=timeout)
            if results:
                return results
        except Exception as exc:
            if _is_timeout_exception(exc):
                _record_search_stat("search_timeouts")
                if fn is search_serper:
                    _record_search_stat("serper_timeout_fallback_to_ddg")
            _record_search_stat("provider_failures")
            errors.append(f"{fn.__name__}: {exc}")

    raise RuntimeError(
        "All configured search providers failed. "
        + (" | ".join(errors) if errors else "No providers were available.")
    )


def run_search(
    query: str,
    cached_search_results: List[str] = None,
    max_search_results_per_query: int = 3,
    max_sentences_per_passage: int = 5,
    sliding_distance: int = 1,
    max_passages_per_search_result_to_return: int = 1,
    timeout: float = 3,
    randomize_num_sentences: bool = False,
    filter_sentence_len: int = 250,
    max_passages_per_search_result_to_score: int = 30,
) -> List[Dict[str, Any]]:
    """Searches the query on a search engine and returns the most relevant information.

    Args:
        query: Search query.
        max_search_results_per_query: Maximum number of search results to get return.
        max_sentences_per_passage: Maximum number of sentences for each passage.
        filter_sentence_len: Maximum length of a sentence before being filtered.
        sliding_distance: Sliding distance over the sentences of each search result.
            Used to extract passages.
        max_passages_per_search_result_to_score: Maxinum number of passages to score for
            each search result.
        max_passages_per_search_result_to_return: Maximum number of passages to return
            for each search result.
    Returns:
        retrieved_passages: Top retrieved passages for the search query.
    """
    mode = _condition_mode()
    if timeout <= 0:
        timeout = 3

    env_timeout = os.getenv("RARR_SEARCH_TIMEOUT", "").strip()
    if env_timeout:
        try:
            timeout = max(1.0, float(env_timeout))
        except Exception:
            pass
    if mode == "structural":
        _record_search_stat("structural_queries_total")

    if _is_missing_text(query):
        _record_search_stat("queries_with_no_results")
        return []

    query = str(query).strip()

    if cached_search_results is not None:
        search_results = cached_search_results
    else:
        raw_retries = os.getenv("RARR_SEARCH_RETRIES", "3").strip()
        try:
            num_retries = max(0, int(raw_retries))
        except Exception:
            num_retries = 3

        search_results = []
        last_error: Exception | None = None
        for attempt in range(num_retries + 1):
            try:
                search_results = search_web(query, timeout=timeout)
                if search_results:
                    break
                last_error = RuntimeError(f"No search results returned for query {query!r}")
            except Exception as exc:
                last_error = exc

            if attempt >= num_retries:
                break

            retry_wait = _search_retry_wait_seconds(attempt + 1)
            logger.warning(
                "[search] Search failed for query %r: %s. Retrying in %.2fs (%d/%d)...",
                query,
                last_error,
                retry_wait,
                attempt + 1,
                num_retries,
            )
            time.sleep(retry_wait)

        if not search_results:
            _record_search_stat("queries_discarded")
            raise SearchQueryFailed(
                f"Search failed for query {query!r} after {num_retries + 1} attempts"
            ) from last_error

    if not search_results:
        _record_search_stat("queries_with_no_results")
        return []

    # Scrape search results in parallel
    with concurrent.futures.ThreadPoolExecutor() as e:
        scraped_results = e.map(scrape_url, search_results, itertools.repeat(timeout))
    # Remove URLs if we weren't able to scrape anything or if they are a PDF.
    scraped_results = [r for r in scraped_results if r[0] and ".pdf" not in r[1]]

    selected_results = scraped_results[:max_search_results_per_query]
    structural_contexts_by_domain: Dict[str, str] = {}
    if mode == "structural":
        structural_contexts_by_domain = _prefetch_structural_contexts(
            [_extract_domain(url) for _, url in selected_results]
        )

    # Iterate through the scraped results and extract out the most useful passages.
    retrieved_passages = []
    saw_structural_context = False
    for webtext, url in selected_results:
        structural_context = ""
        domain = _extract_domain(url)
        if mode == "structural":
            structural_context = structural_contexts_by_domain.get(domain, "")
        elif mode == "third-party":
            structural_context = _lookup_third_party_context(domain)
            _record_search_stat("third_party_evidence_total")
            if structural_context:
                _record_search_stat("third_party_evidence_scored")
        elif mode == "source_attr":
            structural_context = _format_source_attr_context(domain)

        if randomize_num_sentences:
            sents_per_passage = random.randint(1, max_sentences_per_passage)
        else:
            sents_per_passage = max_sentences_per_passage

        # Chunk the extracted text into passages.
        passages = chunk_text(
            text=webtext,
            sentences_per_passage=sents_per_passage,
            filter_sentence_len=filter_sentence_len,
            sliding_distance=sliding_distance,
        )
        passages = passages[:max_passages_per_search_result_to_score]
        if not passages:
            continue

        # Score the passages by relevance to the query using a cross-encoder.
        scores = PASSAGE_RANKER.predict([(query, p) for p in passages]).tolist()
        passage_scores = list(zip(passages, scores))

        # Take the top passages_per_search passages for the current search result.
        passage_scores.sort(key=lambda x: x[1], reverse=True)
        for passage, score in passage_scores[:max_passages_per_search_result_to_return]:
            retrieved_passages.append(
                {
                    "text": passage,
                    "url": url,
                    "query": query,
                    "sents_per_passage": sents_per_passage,
                    "structural_context": structural_context,
                    "retrieval_score": score,  # Cross-encoder score as retr score
                }
            )
            if mode == "structural":
                _record_search_stat("structural_evidence_total")
                if structural_context:
                    _record_search_stat("structural_evidence_with_context")
                    saw_structural_context = True

    if mode == "structural" and saw_structural_context:
        _record_search_stat("structural_queries_with_context")

    if retrieved_passages:
        # Sort all retrieved passages by the retrieval score.
        retrieved_passages = sorted(
            retrieved_passages, key=lambda d: d["retrieval_score"], reverse=True
        )

        # Normalize the retreival scores into probabilities
        scores = [r["retrieval_score"] for r in retrieved_passages]
        probs = torch.nn.functional.softmax(torch.Tensor(scores), dim=-1).tolist()
        for prob, passage in zip(probs, retrieved_passages):
            passage["score"] = prob

    return retrieved_passages
