import logging
import os

from context_core.core.fact_check_state import FactCheckerState
from context_core.core.task_solver import StandardTaskSolver
from context_core.core import register_solver
from context_core.utils.question_generation import run_rarr_question_generation
from context_core.prompts import functional_prompt
from context_core.utils import search


DEFAULT_MAX_CLAIM_CHARS = 2000


def _max_claim_chars() -> int:
    raw = os.getenv("RARR_MAX_CLAIM_CHARS", str(DEFAULT_MAX_CLAIM_CHARS)).strip()
    try:
        return max(240, int(raw))
    except Exception:
        return DEFAULT_MAX_CLAIM_CHARS


def _truncate_claim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "na", "none", "null", "nan", "n/a"}:
        return ""
    return text


@register_solver("rarr_retriever", "claims", "claims_with_evidences")
class RARRRetriever(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.global_config.get("rarr_model", "text-davinci-003")
        self.temperature_qgen = args.get("temperature_qgen", 0.7)
        self.num_rounds_qgen = args.get("num_rounds_qgen", 3)
        self.max_search_results_per_query = args.get("max_search_results_per_query", 5)
        self.max_sentences_per_passage = args.get("max_sentences_per_passage", 4)
        self.sliding_distance = args.get("sliding_distance", 1)
        self.max_passages_per_search_result = args.get("max_passages_per_search_result", 1)

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims_input = state.get(self.input_name)
        if isinstance(claims_input, str):
            cleaned = _clean_text(claims_input)
            claims = [cleaned] if cleaned else []
        elif isinstance(claims_input, list):
            claims = []
            for claim in claims_input:
                cleaned = _clean_text(claim)
                if cleaned:
                    claims.append(cleaned)
        else:
            cleaned = _clean_text(claims_input)
            claims = [cleaned] if cleaned else []

        if not claims:
            state.set(self.output_name, {})
            return True, state
        
        results = dict()
        max_claim_chars = _max_claim_chars()
        for claim in claims:
            claim_for_qgen = _truncate_claim(claim, max_claim_chars)
            try:
                questions = run_rarr_question_generation(
                    claim=claim_for_qgen,
                    context=None,
                    model=self.model,
                    prompt=functional_prompt.QGEN_PROMPT,
                    temperature=self.temperature_qgen,
                    num_rounds=self.num_rounds_qgen,
                )
            except Exception as exc:
                logging.warning(
                    "[rarr_retriever] qgen failed; falling back to claim-as-query. claim_len=%d err=%s",
                    len(claim_for_qgen),
                    exc,
                )
                questions = [claim_for_qgen]
            questions = [q for q in (_clean_text(q) for q in questions) if q]
            if not questions:
                questions = [claim_for_qgen]

            evidences = []
            for question in questions:
                q_evidences = search.run_search(
                    query=question,
                    max_search_results_per_query=self.max_search_results_per_query,
                    max_sentences_per_passage=self.max_sentences_per_passage,
                    sliding_distance=self.sliding_distance,
                    max_passages_per_search_result_to_return=self.max_passages_per_search_result,
                )
                evidences.extend(
                    [
                        (question, x["text"], x.get("structural_context", ""))
                        for x in q_evidences
                    ]
                )
               
            results[claim] = evidences

        state.set(self.output_name, results)
        return True, state
