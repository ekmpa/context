import logging

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

    def __call__(self, state: FactCheckerState, *args, **kwargs):
        claims_with_evidences = state.get(self.input_name)
        results = []
        for claim, evidences in claims_with_evidences.items():
            result = {}
            evidences = evidences[:self.max_evidences_per_question]
            decisions = []
            for evidence_item in evidences:
                if len(evidence_item) >= 3:
                    query, evidence, structural_context = evidence_item[0], evidence_item[1], evidence_item[2]
                else:
                    query, evidence = evidence_item
                    structural_context = ""
                gate = agreement_gate.run_agreement_gate(
                    claim=claim,
                    context=None,
                    query=query,
                    evidence=evidence,
                    structural_context=structural_context,
                    model=self.model,
                    prompt=functional_prompt.AGREEMENT_GATE_PROMPT
                )
                decisions.append(gate["decision"])
            result['claim'] = claim
            result['evidences'] = evidences
            result['labels'] = decisions

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
