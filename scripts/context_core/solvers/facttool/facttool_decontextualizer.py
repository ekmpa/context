import ast

from context_core.core import FactCheckerState, StandardTaskSolver, register_solver
from context_core.llm import chat_text

from .prompts import CLAIM_EXTRACTION_SYSTEM_PROMPT, CLAIM_EXTRACTION_USER_PROMPT


@register_solver("facttool_decontextualizer", "response", "claims")
class FactToolDecontextualizer(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.model = self.global_config.get("llm_in_use", "gpt-4o-mini")
        self.num_retries = int(self.global_config.get("num_retries", 3))

    def _parse_claims(self, raw: str) -> list[str]:
        try:
            payload = ast.literal_eval(raw)
        except Exception:
            return []

        if not isinstance(payload, list):
            return []

        claims: list[str] = []
        for item in payload:
            if isinstance(item, dict):
                text = str(item.get("claim", "")).strip()
                if text:
                    claims.append(text)
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    claims.append(text)
        return claims

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        response = str(state.get(self.input_name) or "").strip()
        if not response:
            state.set(self.output_name, [])
            return True, state

        prompt = CLAIM_EXTRACTION_USER_PROMPT.replace("{input}", response)
        raw = chat_text(
            prompt,
            model=self.model,
            system_role=CLAIM_EXTRACTION_SYSTEM_PROMPT,
            num_retries=self.num_retries,
        )
        claims = self._parse_claims(raw)
        if not claims:
            # Fallback keeps pipeline running even when output format drifts.
            claims = [response]

        state.set(self.output_name, claims)
        return True, state
