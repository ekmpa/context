import ast
import logging

from context_core.core import FactCheckerState, StandardTaskSolver, register_solver
from context_core.llm import chat_text
from context_core.utils import search

from .prompts import QUERY_GENERATION_SYSTEM_PROMPT, QUERY_GENERATION_USER_PROMPT


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "na", "none", "null", "nan", "n/a"}:
        return ""
    return text


@register_solver("facttool_evidence_retriever", "claims", "claims_with_evidences")
class FactToolEvidenceRetriever(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.global_config.get("llm_in_use", "gpt-4o-mini")
        self.num_retries = int(self.global_config.get("num_retries", 3))
        self.max_search_results_per_query = int(args.get("max_search_results_per_query", 5))
        self.max_sentences_per_passage = int(args.get("max_sentences_per_passage", 4))
        self.sliding_distance = int(args.get("sliding_distance", 1))
        self.max_passages_per_search_result = int(args.get("max_passages_per_search_result", 1))

    def _generate_queries(self, claim: str) -> list[str]:
        prompt = QUERY_GENERATION_USER_PROMPT.replace("{input}", claim)
        raw = chat_text(
            prompt,
            model=self.model,
            system_role=QUERY_GENERATION_SYSTEM_PROMPT,
            num_retries=self.num_retries,
        )
        try:
            parsed = ast.literal_eval(raw)
        except Exception:
            return [claim]

        if not isinstance(parsed, list):
            return [claim]

        queries = [q for q in (_clean_text(v) for v in parsed) if q]
        return queries or [claim]

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims = state.get(self.input_name)
        if not isinstance(claims, list):
            claims = [_clean_text(claims)] if _clean_text(claims) else []
        else:
            claims = [c for c in (_clean_text(x) for x in claims) if c]

        if not claims:
            state.set(self.output_name, {})
            return True, state

        search_query_failures = 0
        search_queries_succeeded = 0
        total_queries = 0
        results: dict[str, list[tuple[str, str, str]]] = {}

        for claim in claims:
            queries = self._generate_queries(claim)
            evidences = []
            for query in queries:
                total_queries += 1
                try:
                    q_evidences = search.run_search(
                        query=query,
                        max_search_results_per_query=self.max_search_results_per_query,
                        max_sentences_per_passage=self.max_sentences_per_passage,
                        sliding_distance=self.sliding_distance,
                        max_passages_per_search_result_to_return=self.max_passages_per_search_result,
                    )
                except search.SearchQueryFailed as exc:
                    search_query_failures += 1
                    logging.warning(
                        "[facttool_evidence_retriever] search failed; skipping query=%r err=%s",
                        query,
                        exc,
                    )
                    continue

                search_queries_succeeded += 1
                evidences.extend(
                    [(query, x["text"], x.get("structural_context", "")) for x in q_evidences]
                )

            results[claim] = evidences

        state.set("search_queries_total", total_queries)
        state.set("search_queries_succeeded", search_queries_succeeded)
        state.set("search_query_failures", search_query_failures)
        state.set(
            "search_failed",
            total_queries > 0 and search_queries_succeeded == 0 and search_query_failures > 0,
        )
        state.set(self.output_name, results)
        return True, state
