import os
import unittest

from scripts.llm_utils import query_gemini, query_gpt


def _is_enabled() -> bool:
    return os.getenv("RUN_LIVE_CLOSED_SOURCE_TESTS", "0").strip() == "1"


@unittest.skipUnless(_is_enabled(), "Set RUN_LIVE_CLOSED_SOURCE_TESTS=1 to run live API tests")
class TestClosedSourceLive(unittest.TestCase):
    def test_query_gpt_live(self):
        model = os.getenv("OPENAI_TEST_MODEL", "gpt-4.1-mini")
        text = query_gpt(model, "Reply with exactly: pong")
        self.assertTrue(text)

    def test_query_gemini_live(self):
        model = os.getenv("GEMINI_TEST_MODEL", "gemini-2.0-flash")
        text = query_gemini(model, "Reply with exactly: pong")
        self.assertTrue(text)


if __name__ == "__main__":
    unittest.main()
