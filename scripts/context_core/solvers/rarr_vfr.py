import logging
import os
import concurrent.futures

from context_core.core.fact_check_state import FactCheckerState
from context_core.core.task_solver import StandardTaskSolver
from context_core.core import register_solver
from context_core.utils import agreement_gate
from context_core.prompts import functional_prompt


@register_solver("rarr_verifier", "claims_with_evidences", "label")
class RARRAgreementGate(StandardTaskSolver):
    def __init__(self, args):
        super().__init__(args)
        self.max_evidences_per_question = args.get("max_evidences_per_question", 1)
        self.model = self.global_config.get("rarr_model", "text-davinci-003")
        raw_workers = str(args.get("gate_workers", os.getenv("RARR_GATE_WORKERS", "1"))).strip()
        try:
            self.gate_workers = max(1, int(raw_workers))
        except Exception:
            self.gate_workers = 1

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims_with_evidences = state.get(self.input_name)
        results = []
        for claim, evidences in claims_with_evidences.items():
            result = {}
            evidences = evidences[:self.max_evidences_per_question]
            decisions = []
            gate_debug = []

            def _evaluate_evidence(evidence_item):
                if len(evidence_item) >= 3:
                    query, evidence, structural_context = evidence_item[0], evidence_item[1], evidence_item[2]
                else:
                    query, evidence = evidence_item
                    structural_context = ""
                try:
                    gate = agreement_gate.run_agreement_gate(
                        claim=claim,
                        context=None,
                        query=query,
                        evidence=evidence,
                        structural_context=structural_context,
                        model=self.model,
                        prompt=functional_prompt.AGREEMENT_GATE_PROMPT
                    )
                except Exception as exc:
                    logging.warning("[rarr_verifier] agreement gate failed; marking evidence unverifiable: %s", exc)
                    gate = {
                        "is_open": False,
                        "reason": f"agreement gate failed: {exc}",
                        "decision": "unverifiable",
                        "raw_response": "",
                        "prompt_input": "",
                    }
                return {
                    "decision": gate["decision"],
                    "debug": {
                        "query": query,
                        "decision": gate.get("decision"),
                        "reason": gate.get("reason"),
                        "raw_response": gate.get("raw_response", ""),
                        "prompt_input": gate.get("prompt_input", ""),
                    },
                }

            if self.gate_workers <= 1 or len(evidences) <= 1:
                for evidence_item in evidences:
                    gate_out = _evaluate_evidence(evidence_item)
                    decisions.append(gate_out["decision"])
                    gate_debug.append(gate_out["debug"])
            else:
                max_workers = min(self.gate_workers, len(evidences))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = [pool.submit(_evaluate_evidence, evidence_item) for evidence_item in evidences]
                    for fut in futures:
                        gate_out = fut.result()
                        decisions.append(gate_out["decision"])
                        gate_debug.append(gate_out["debug"])

            result['claim'] = claim
            result['evidences'] = evidences
            result['labels'] = decisions
            result['gate_debug'] = gate_debug

            if decisions and all(d == "agrees" for d in decisions):
                result['factuality'] = True
            elif decisions and all(d == "disagrees" for d in decisions):
                result['factuality'] = False
            else:
                result['factuality'] = None
            results.append(result)
        state.set(self.output_name, all(x.get('factuality') is True for x in results))
        state.set("detail", results)
        return True, state
