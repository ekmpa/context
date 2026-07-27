import ast
import json
import re

from context_core.core import FactCheckerState, StandardTaskSolver, register_solver
from context_core.llm import chat_text

from .prompts import VERIFICATION_SYSTEM_PROMPT, VERIFICATION_USER_PROMPT


def _parse_verification(raw: str) -> dict:
    text = str(raw or "").strip()
    if not text:
        return {}

    # Strip optional thinking blocks that some local models emit.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # Keep only the outermost dictionary-like span if extra text is present.
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        text = text[first : last + 1].strip()

    # Some generations wrap dicts with doubled braces (e.g. {{ ... }}).
    if text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1].strip()

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        # Fallback for JSON-like output; normalize booleans first.
        normalized = re.sub(r"\bTrue\b", "true", text)
        normalized = re.sub(r"\bFalse\b", "false", normalized)
        normalized = re.sub(r"\bNone\b", "null", normalized)
        try:
            parsed = json.loads(normalized)
        except Exception:
            return {}

    if isinstance(parsed, dict):
        factuality = parsed.get("factuality")
        if isinstance(factuality, bool):
            return parsed
    return {}


def _to_label(factuality):
    if factuality is True:
        return "agrees"
    if factuality is False:
        return "disagrees"
    return "unverifiable"


@register_solver("facttool_claim_examiner", "claims_with_evidences", "label")
class FactToolClaimExaminer(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.global_config.get("llm_in_use", "gpt-4o-mini")
        self.num_retries = int(self.global_config.get("num_retries", 3))
        self.max_evidences_per_claim = int(args.get("max_evidences_per_claim", 5))

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims_with_evidences = state.get(self.input_name)
        if not isinstance(claims_with_evidences, dict):
            state.set("detail", [])
            state.set(self.output_name, False)
            return True, state

        details = []
        all_true = True

        for claim, evidences in claims_with_evidences.items():
            claim_text = str(claim or "").strip()
            rows = evidences if isinstance(evidences, list) else []
            rows = rows[: self.max_evidences_per_claim]

            evidence_text = []
            for row in rows:
                if isinstance(row, (list, tuple)) and len(row) > 1:
                    evidence_text.append(str(row[1]))
                elif isinstance(row, dict):
                    evidence_text.append(str(row.get("text", "")))

            prompt = VERIFICATION_USER_PROMPT.replace("{claim}", claim_text).replace(
                "{evidence}", str(evidence_text)
            )
            raw = chat_text(
                prompt,
                model=self.model,
                system_role=VERIFICATION_SYSTEM_PROMPT,
                num_retries=self.num_retries,
            )
            parsed = _parse_verification(raw)
            factuality = parsed.get("factuality")
            label = _to_label(factuality)
            if factuality is not True:
                all_true = False

            details.append(
                {
                    "claim": claim_text,
                    "evidences": rows,
                    "labels": [label],
                    "factuality": factuality if isinstance(factuality, bool) else None,
                    "gate_debug": [
                        {
                            "decision": label,
                            "raw_response": raw,
                            "prompt_input": prompt,
                            "reason": str(parsed.get("reasoning", "")),
                        }
                    ],
                }
            )

        state.set("detail", details)
        state.set(self.output_name, all_true and bool(details))
        return True, state
