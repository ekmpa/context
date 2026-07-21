import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from context_core.utils.agreement_gate import parse_api_response


class TestAgreementGateParser(unittest.TestCase):
    def test_uses_final_label_when_prompt_is_echoed(self):
        response = (
            "Return exactly one label:\n"
            "agrees | disagrees | ambiguous | unverifiable\n"
            "assistant\n"
            "<think>analysis</think>\n"
            "disagrees"
        )
        is_open, reason, decision = parse_api_response(response)
        self.assertFalse(is_open)
        self.assertEqual(decision, "disagrees")
        self.assertTrue(reason)

    def test_label_line_is_still_parsed(self):
        response = "Reasoning: evidence conflicts.\nLabel: disagrees"
        is_open, _reason, decision = parse_api_response(response)
        self.assertFalse(is_open)
        self.assertEqual(decision, "disagrees")


if __name__ == "__main__":
    unittest.main()
