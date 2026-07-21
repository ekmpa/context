"""Utils for running the agreement gate."""
import os
import re
from typing import Any, Dict, Tuple

from context_core.llm import chat_text, completion_text, chat_text_batch, completion_text_batch


DEFAULT_MAX_GATE_CLAIM_CHARS = 2000
DEFAULT_MAX_GATE_QUERY_CHARS = 400
DEFAULT_MAX_GATE_EVIDENCE_CHARS = 3500
DEFAULT_MAX_GATE_STRUCTURAL_CHARS = 1000


COMPACT_AGREEMENT_GATE_TEMPLATE = """You are a strict fact-checking judge.

Claim: {claim}
Search query: {query}
Evidence snippet: {evidence}
Source/domain context: {structural_context}

If the source/domain context is temporal (history across months), consider historical patterns in the source assessment.

Return exactly one label from:
agrees | disagrees | ambiguous | unverifiable
""".strip()

RAW_COMPACT_GATE_TEMPLATE = """Claim: {claim}
Search query: {query}
Evidence: {evidence}

Return exactly one label:
agrees | disagrees | ambiguous | unverifiable
""".strip()


def _get_int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return max(minimum, int(raw))
    except Exception:
        return default


def _clip_text(value: object, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


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
    response_text = (api_response or "").strip()
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
    reason = ""

    reasoning_line = next((line for line in lines if line.lower().startswith("reasoning:")), "")
    if reasoning_line:
        reason = reasoning_line.split(":", 1)[-1].strip()
    elif lines:
        reason = lines[0]

    decision_matches = re.findall(
        r"(?:therefore\s*:|final answer\s*:|decision\s*:|label\s*:)?\s*"
        r"(agrees|disagrees|ambiguous|unverifiable|irrelevant|unknown|support|supports|refute|refutes|contradict|contradicts)\b",
        response_text,
        flags=re.IGNORECASE,
    )
    if decision_matches:
        # Some local models echo the full prompt (including "agrees | disagrees ...")
        # before their answer. Use the last matched label token to capture the
        # model's final decision instead of an echoed instruction token.
        decision = _normalize_decision(decision_matches[-1])
    elif lines:
        decision = _normalize_decision(lines[-1])
    else:
        decision = "unverifiable"
        if not reason:
            reason = "Failed to parse."

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
    claim = _clip_text(
        claim,
        _get_int_env("RARR_MAX_GATE_CLAIM_CHARS", DEFAULT_MAX_GATE_CLAIM_CHARS, 240),
    )
    query = _clip_text(
        query,
        _get_int_env("RARR_MAX_GATE_QUERY_CHARS", DEFAULT_MAX_GATE_QUERY_CHARS, 64),
    )
    evidence = _clip_text(
        evidence,
        _get_int_env("RARR_MAX_GATE_EVIDENCE_CHARS", DEFAULT_MAX_GATE_EVIDENCE_CHARS, 256),
    )
    structural_context = _clip_text(
        structural_context,
        _get_int_env("RARR_MAX_GATE_STRUCTURAL_CHARS", DEFAULT_MAX_GATE_STRUCTURAL_CHARS, 128),
    )

    use_compact_prompt = os.getenv("RARR_USE_COMPACT_GATE_PROMPT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if use_compact_prompt:
        if structural_context:
            gpt3_input = COMPACT_AGREEMENT_GATE_TEMPLATE.format(
                claim=claim,
                query=query,
                evidence=evidence,
                structural_context=structural_context,
            ).strip()
        else:
            gpt3_input = RAW_COMPACT_GATE_TEMPLATE.format(
                claim=claim,
                query=query,
                evidence=evidence,
            ).strip()
    elif context:
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

    if use_compact_prompt:
        system_role = (
            "You are a fact-checking judge. Output exactly one label: agrees, disagrees, ambiguous, or unverifiable."
            if not structural_context
            else (
                "You are a strict fact-checking judge. "
                "Use source/domain context as a reliability prior. "
                "When temporal history is present, weigh consistency across months and repeated peers as stronger evidence than one-off spikes. "
                "Output exactly one token label: agrees, disagrees, ambiguous, or unverifiable."
            )
        )
        response_text = chat_text(
            [{"role": "user", "content": gpt3_input}],
            model=model,
            system_role=system_role,
            temperature=0.0,
            num_retries=num_retries,
            waiting=2.0,
        )
    else:
        response_text = completion_text(
            gpt3_input,
            model=model,
            temperature=0.0,
            max_tokens=256,
            num_retries=num_retries,
            waiting=2.0,
            logit_bias={"50256": -100},
        )

    if not (response_text or "").strip():
        fallback_input = (
            gpt3_input
            + "\n\nRespond with exactly one line in this format: "
            + "agrees|disagrees|ambiguous|unverifiable"
        )
        if use_compact_prompt:
            response_text = chat_text(
                [{"role": "user", "content": fallback_input}],
                model=model,
                system_role="Output one label only: agrees, disagrees, ambiguous, or unverifiable.",
                temperature=0.0,
                num_retries=max(1, num_retries),
                waiting=2.0,
            )
        else:
            response_text = completion_text(
                fallback_input,
                model=model,
                temperature=0.0,
                max_tokens=64,
                num_retries=max(1, num_retries),
                waiting=2.0,
                logit_bias={"50256": -100},
            )

    is_open, reason, decision = parse_api_response(response_text)
    gate = {
        "is_open": is_open,
        "reason": reason,
        "decision": decision,
        "raw_response": response_text,
        "prompt_input": gpt3_input,
    }
    return gate


def run_agreement_gate_batch(
    claim: str,
    evidence_rows: list[tuple[str, str, str]],
    model: str,
    prompt: str,
    context: str = None,
    num_retries: int = 5,
) -> list[Dict[str, Any]]:
    if not evidence_rows:
        return []

    prepared: list[tuple[str, str]] = []
    claim_clipped = _clip_text(
        claim,
        _get_int_env("RARR_MAX_GATE_CLAIM_CHARS", DEFAULT_MAX_GATE_CLAIM_CHARS, 240),
    )
    use_compact_prompt = os.getenv("RARR_USE_COMPACT_GATE_PROMPT", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    for query, evidence, structural_context in evidence_rows:
        query_clipped = _clip_text(
            query,
            _get_int_env("RARR_MAX_GATE_QUERY_CHARS", DEFAULT_MAX_GATE_QUERY_CHARS, 64),
        )
        evidence_clipped = _clip_text(
            evidence,
            _get_int_env("RARR_MAX_GATE_EVIDENCE_CHARS", DEFAULT_MAX_GATE_EVIDENCE_CHARS, 256),
        )
        structural_clipped = _clip_text(
            structural_context,
            _get_int_env("RARR_MAX_GATE_STRUCTURAL_CHARS", DEFAULT_MAX_GATE_STRUCTURAL_CHARS, 128),
        )

        if use_compact_prompt:
            if structural_clipped:
                gpt3_input = COMPACT_AGREEMENT_GATE_TEMPLATE.format(
                    claim=claim_clipped,
                    query=query_clipped,
                    evidence=evidence_clipped,
                    structural_context=structural_clipped,
                ).strip()
            else:
                gpt3_input = RAW_COMPACT_GATE_TEMPLATE.format(
                    claim=claim_clipped,
                    query=query_clipped,
                    evidence=evidence_clipped,
                ).strip()
            system_role = (
                "You are a fact-checking judge. Output exactly one label: agrees, disagrees, ambiguous, or unverifiable."
                if not structural_clipped
                else (
                    "You are a strict fact-checking judge. "
                    "Use source/domain context as a reliability prior. "
                    "When temporal history is present, weigh consistency across months and repeated peers as stronger evidence than one-off spikes. "
                    "Output exactly one token label: agrees, disagrees, ambiguous, or unverifiable."
                )
            )
        elif context:
            gpt3_input = prompt.format(
                context=context,
                claim=claim_clipped,
                query=query_clipped,
                evidence=evidence_clipped,
                structural_context=structural_clipped or "No structural graph evidence provided.",
            ).strip()
            system_role = ""
        else:
            gpt3_input = prompt.format(
                claim=claim_clipped,
                query=query_clipped,
                evidence=evidence_clipped,
                structural_context=structural_clipped or "No structural graph evidence provided.",
            ).strip()
            system_role = ""

        prepared.append((gpt3_input, system_role))

    if use_compact_prompt:
        # When system role differs per prompt, issue one grouped call per system role.
        grouped: dict[str, list[tuple[int, str]]] = {}
        for idx, (gpt3_input, system_role) in enumerate(prepared):
            grouped.setdefault(system_role, []).append((idx, gpt3_input))

        raw_outputs: list[str] = [""] * len(prepared)
        for system_role, entries in grouped.items():
            batch_inputs = [[{"role": "user", "content": prompt_input}] for _, prompt_input in entries]
            batch_outputs = chat_text_batch(
                batch_inputs,
                model=model,
                system_role=system_role,
                temperature=0.0,
                num_retries=num_retries,
                waiting=2.0,
            )
            for (idx, _), text in zip(entries, batch_outputs):
                raw_outputs[idx] = text
    else:
        prompts = [gpt3_input for gpt3_input, _ in prepared]
        raw_outputs = completion_text_batch(
            prompts,
            model=model,
            temperature=0.0,
            max_tokens=256,
            num_retries=num_retries,
            waiting=2.0,
            logit_bias={"50256": -100},
        )

    gates: list[Dict[str, Any]] = []
    for (gpt3_input, _), response_text in zip(prepared, raw_outputs):
        response_text = (response_text or "").strip()
        if not response_text:
            response_text = "unverifiable"
        is_open, reason, decision = parse_api_response(response_text)
        gates.append(
            {
                "is_open": is_open,
                "reason": reason,
                "decision": decision,
                "raw_response": response_text,
                "prompt_input": gpt3_input,
            }
        )
    return gates
