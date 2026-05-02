"""Utils for searching a query and returning top passages from search results."""
import concurrent.futures
import csv
import itertools
import json
import logging
import os
import random
import subprocess
import sys
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
SEARCH_STATS: Dict[str, int] = {
    "search_timeouts": 0,
    "serper_timeout_fallback_to_ddg": 0,
    "provider_failures": 0,
    "queries_with_no_results": 0,
}
THIRD_PARTY_RATINGS: Dict[str, Dict[str, float]] = {}
THIRD_PARTY_RATINGS_LOADED = False
logger = logging.getLogger(__name__)


def _record_search_stat(key: str, value: int = 1) -> None:
    SEARCH_STATS[key] = SEARCH_STATS.get(key, 0) + value


def reset_search_stats() -> None:
    for key in list(SEARCH_STATS.keys()):
        SEARCH_STATS[key] = 0


def get_search_stats() -> Dict[str, int]:
    return dict(SEARCH_STATS)


def _is_timeout_exception(exc: Exception) -> bool:
    return isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ReadTimeout))


def _structural_enabled() -> bool:
    return os.getenv("RARR_STRUCTURAL_MODE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _condition_mode() -> str:
    mode = os.getenv("RARR_CONDITION", "").strip().lower()
    if mode in {"raw", "structural", "third-party", "third_party", "source_attr", "source-attr"}:
        if mode == "third_party":
            return "third-party"
        if mode in {"source_attr", "source-attr"}:
            return "source_attr"
        return mode
    if _structural_enabled():
        return "structural"
    return "raw"


def _is_missing_text(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "na", "none", "null", "nan", "n/a"}


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").split(":")[0].strip().lower()
    if not host:
        # Allow plain domains without scheme, e.g. "reuters.com".
        host = (parsed.path or "").split("/")[0].split(":")[0].strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host


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

    if isinstance(payload, list):
        neighbors = [x for x in payload if isinstance(x, str)][:max_neighbors]
        if not neighbors:
            return f"Domain graph hints for {domain}: no recent neighbors found in hook output."
        return (
            f"Domain graph hints for {domain}. "
            f"Latest-graph {os.getenv('RARR_STRUCTURAL_HOPS', '2')}-hop neighbors: "
            + ", ".join(neighbors)
            + "."
        )

    if not isinstance(payload, dict):
        return f"Domain graph hints for {domain}: hook output format was not recognized."

    month_counts = [
        f"{month}:{len(payload.get(month, []))}"
        for month in STRUCTURAL_MONTH_ORDER
        if payload.get(month)
    ]

    most_recent_neighbors = []
    most_recent_month = None
    for month in STRUCTURAL_MONTH_ORDER:
        if payload.get(month):
            most_recent_month = month
            most_recent_neighbors = payload.get(month, [])[:max_neighbors]
            break

    if not month_counts and not most_recent_neighbors:
        return f"Domain graph hints for {domain}: no neighbor data found in hook index."

    summary = f"Domain graph hints for {domain}. Monthly neighbor counts: {'; '.join(month_counts)}."
    if most_recent_month and most_recent_neighbors:
        summary += (
            f" Example neighbors from {most_recent_month}: "
            + ", ".join(most_recent_neighbors)
            + "."
        )
    return summary


def _lookup_structural_context(domain: str, timeout: float = 6.0) -> str:
    if not domain:
        return ""

    _load_structural_cache_file()
    if domain in STRUCTURAL_CACHE:
        return STRUCTURAL_CACHE[domain]

    hook_path = os.getenv("RARR_STRUCTURAL_HOOK_PATH", "").strip()
    if not hook_path:
        hook_path = str(Path(__file__).resolve().parents[2] / "hook.py")
    if not os.path.isfile(hook_path):
        return ""

    try:
        cmd = [
            sys.executable,
            hook_path,
            domain,
            "--mode",
            "latest",
            "--hops",
            os.getenv("RARR_STRUCTURAL_HOPS", "2"),
            "--max-domains-per-hop",
            os.getenv("RARR_STRUCTURAL_MAX_DOMAINS_PER_HOP", "10"),
        ]

        shards_dir = os.getenv("RARR_STRUCTURAL_SHARDS_DIR", "").strip()
        if shards_dir:
            cmd.extend(["--serving-shards-dir", shards_dir])

        raw = subprocess.check_output(
            cmd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
        payload = json.loads(raw)
        summary = _format_structural_context(domain, payload)
        STRUCTURAL_CACHE[domain] = summary
        _persist_structural_cache_file()
        return summary
    except Exception:
        STRUCTURAL_CACHE[domain] = ""
        _persist_structural_cache_file()
        return ""


def _get_structural_cache_file() -> str:
    return os.getenv("RARR_STRUCTURAL_CACHE_FILE", "").strip()


def _load_structural_cache_file() -> None:
    global STRUCTURAL_CACHE_LOADED
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


def _persist_structural_cache_file() -> None:
    global STRUCTURAL_CACHE_DIRTY
    cache_file = _get_structural_cache_file()
    if not cache_file:
        return

    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(STRUCTURAL_CACHE, handle, ensure_ascii=True, sort_keys=True)
        STRUCTURAL_CACHE_DIRTY = False
    except Exception:
        STRUCTURAL_CACHE_DIRTY = True


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
    """Dispatch web search by provider with an automatic fallback policy.

    Providers:
    - serper: requires SERPER_API_KEY
    - duckduckgo: no key
    - auto: serper -> duckduckgo
    """
    provider = os.getenv("RARR_SEARCH_PROVIDER", "serper").strip().lower()

    if provider == "serper":
        try:
            return search_serper(query, timeout=timeout)
        except Exception as exc:
            if _is_timeout_exception(exc):
                _record_search_stat("search_timeouts")
                _record_search_stat("serper_timeout_fallback_to_ddg")
                logger.warning(
                    "[search] Serper timed out for query %r; trying DuckDuckGo instead.",
                    query,
                )
            else:
                _record_search_stat("provider_failures")
                logger.warning(
                    "[search] Serper failed for query %r: %s. Trying DuckDuckGo instead.",
                    query,
                    exc,
                )

            try:
                return search_duckduckgo(query, timeout=timeout)
            except Exception as ddg_exc:
                if _is_timeout_exception(ddg_exc):
                    _record_search_stat("search_timeouts")
                _record_search_stat("provider_failures")
                logger.warning(
                    "[search] DuckDuckGo failed after Serper failure for query %r: %s",
                    query,
                    ddg_exc,
                )
                return []
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
    if _is_missing_text(query):
        _record_search_stat("queries_with_no_results")
        return []

    query = str(query).strip()

    if cached_search_results is not None:
        search_results = cached_search_results
    else:
        try:
            search_results = search_web(query, timeout=timeout)
        except Exception as exc:
            _record_search_stat("provider_failures")
            logger.warning("[search] Search failed for query %r: %s", query, exc)
            return []

    if not search_results:
        _record_search_stat("queries_with_no_results")
        return []

    # Scrape search results in parallel
    with concurrent.futures.ThreadPoolExecutor() as e:
        scraped_results = e.map(scrape_url, search_results, itertools.repeat(timeout))
    # Remove URLs if we weren't able to scrape anything or if they are a PDF.
    scraped_results = [r for r in scraped_results if r[0] and ".pdf" not in r[1]]

    # Iterate through the scraped results and extract out the most useful passages.
    retrieved_passages = []
    mode = _condition_mode()
    for webtext, url in scraped_results[:max_search_results_per_query]:
        structural_context = ""
        domain = _extract_domain(url)
        if mode == "structural":
            structural_context = _lookup_structural_context(domain, timeout=timeout)
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
