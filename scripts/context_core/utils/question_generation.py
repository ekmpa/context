"""Utils for running question generation."""
import os
from typing import List

from context_core.llm import completion_text


DEFAULT_MAX_CLAIM_CHARS = 2000
MIN_CLAIM_CHARS = 240


def _max_claim_chars() -> int:
    raw = os.getenv("RARR_MAX_CLAIM_CHARS", str(DEFAULT_MAX_CLAIM_CHARS)).strip()
    try:
        return max(MIN_CLAIM_CHARS, int(raw))
    except Exception:
        return DEFAULT_MAX_CLAIM_CHARS


def _truncate_claim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _is_context_window_error(exc: Exception) -> bool:
    return "context window" in str(exc).lower() or "context length" in str(exc).lower()


def parse_api_response(api_response: str) -> List[str]:
    """Extract questions from the GPT-3 API response.

    Our prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Question generation response from GPT-3.
    Returns:
        questions: A list of questions.
    """
    search_string = "I googled:"
    questions = []
    for question in api_response.split("\n"):
        # Remove the search string from each question
        if search_string not in question:
            continue
        question = question.split(search_string)[1].strip()
        questions.append(question)

    return questions


def run_rarr_question_generation(
    claim: str,
    model: str,
    prompt: str,
    temperature: float,
    num_rounds: int,
    context: str = None,
    num_retries: int = 5,
) -> List[str]:
    """Generates questions that interrogate the information in a claim.

    Given a piece of text (claim), we use GPT-3 to generate questions that question the
    information in the claim. We run num_rounds of sampling to get a diverse set of questions.

    Args:
        claim: Text to generate questions off of.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        temperature: Temperature to use for sampling questions. 0 represents greedy deconding.
        num_rounds: Number of times to sample questions.
    Returns:
        questions: A list of questions.
    """
    claim_text = str(claim or "").strip()
    max_chars = _max_claim_chars()
    truncated_claim = _truncate_claim(claim_text, max_chars)

    def _build_prompt(active_claim: str) -> str:
        if context:
            return prompt.format(context=context, claim=active_claim).strip()
        return prompt.format(claim=active_claim).strip()

    questions = set()
    for _ in range(num_rounds):
        active_claim = truncated_claim
        response_text = ""

        # Retry with smaller claim slices if the model still reports context overflow.
        for _attempt in range(3):
            gpt3_input = _build_prompt(active_claim)
            try:
                response_text = completion_text(
                    gpt3_input,
                    model=model,
                    temperature=temperature,
                    max_tokens=256,
                    num_retries=num_retries,
                )
                break
            except RuntimeError as exc:
                if _is_context_window_error(exc) and len(active_claim) > MIN_CLAIM_CHARS:
                    next_len = max(MIN_CLAIM_CHARS, int(len(active_claim) * 0.6))
                    active_claim = _truncate_claim(active_claim, next_len)
                    continue
                raise

        cur_round_questions = parse_api_response(response_text.strip())
        questions.update(cur_round_questions)

    questions = list(sorted(questions))
    return questions
