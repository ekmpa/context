"""Utils for running the agreement gate."""
from typing import Any, Dict, Tuple

from context_core.llm import completion_text


def _normalize_decision(raw_decision: str | None) -> str:
    if not raw_decision:
        return "unverifiable"
    text = raw_decision.strip().lower()
    if "ambiguous" in text:
        return "ambiguous"
    if "unverifiable" in text or "cannot be verified" in text or "insufficient" in text:
        return "unverifiable"
    if "disagrees" in text or "refute" in text or "contradict" in text:
        return "disagrees"
    if "agrees" in text or "support" in text:
        return "agrees"
    if "irrelevant" in text or "unknown" in text:
        return "unverifiable"
    return "unverifiable"


def parse_api_response(api_response: str) -> Tuple[bool, str, str]:
    """Extract the agreement gate state and the reasoning from the GPT-3 API response.

    Our prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Agreement gate response from GPT-3.
    Returns:
        is_open: Whether the agreement gate is open.
        reason: The reasoning for why the agreement gate is open or closed.
        decision: The decision of the status of the gate in string form.
    """
    lines = [line.strip() for line in api_response.strip().split("\n") if line.strip()]
    if len(lines) < 2:
        reason = "Failed to parse."
        decision = "unverifiable"
        is_open = False
    else:
        reason = lines[0]
        decision_line = next((line for line in lines if "therefore:" in line.lower()), lines[1])
        decision_raw = decision_line.split(":", 1)[-1].strip() if ":" in decision_line else decision_line
        decision = _normalize_decision(decision_raw)
        is_open = decision == "agrees"
    return is_open, reason, decision


def run_agreement_gate(
    claim: str,
    query: str,
    evidence: str,
    structural_context: str,
    model: str,
    prompt: str,
    context: str = None,
    num_retries: int = 5,
) -> Dict[str, Any]:
    """Checks if a provided evidence contradicts the claim given a query.

    Checks if the answer to a query using the claim contradicts the answer using the
    evidence. If so, we open the agreement gate, which means that we allow the editor
    to edit the claim. Otherwise the agreement gate is closed.

    Args:
        claim: Text to check the validity of.
        query: Query to guide the validity check.
        evidence: Evidence to judge the validity of the claim against.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        gate: A dictionary with the status of the gate and reasoning for decision.
    """
    if context:
        gpt3_input = prompt.format(
            context=context,
            claim=claim,
            query=query,
            evidence=evidence,
            structural_context=structural_context or "No structural graph evidence provided.",
        ).strip()
    else:
        gpt3_input = prompt.format(
            claim=claim,
            query=query,
            evidence=evidence,
            structural_context=structural_context or "No structural graph evidence provided.",
        ).strip()

    response_text = completion_text(
        gpt3_input,
        model=model,
        temperature=0.0,
        max_tokens=256,
        stop=["\n\n"],
        num_retries=num_retries,
        waiting=2.0,
        logit_bias={"50256": -100},
    )

    is_open, reason, decision = parse_api_response(response_text)
    gate = {"is_open": is_open, "reason": reason, "decision": decision}
    return gate
