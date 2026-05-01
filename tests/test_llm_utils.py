import unittest
from unittest.mock import patch

from scripts import llm_utils


class _TokenizerWithTemplate:
    eos_token_id = 7

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.last_messages = messages
        self.last_tokenize = tokenize
        self.last_add_generation_prompt = add_generation_prompt
        return f"TEMPLATE::{messages[0]['content']}"


class _TokenizerNoTemplate:
    eos_token_id = 9


class _Generator:
    def __init__(self, outputs):
        self.outputs = outputs
        self.last_args = None
        self.last_kwargs = None

    def __call__(self, *args, **kwargs):
        self.last_args = args
        self.last_kwargs = kwargs
        return self.outputs


class _FakeOpenAIResponses:
    def __init__(self, text):
        self._text = text
        self.last_model = None
        self.last_input = None

    def create(self, model, input):
        self.last_model = model
        self.last_input = input

        class _Resp:
            output_text = ""

        resp = _Resp()
        resp.output_text = self._text
        return resp


class _FakeOpenAIClient:
    def __init__(self, text):
        class _Responses:
            pass

        self.responses = _Responses()
        self.responses.create = _FakeOpenAIResponses(text).create


class _FakeGeminiModels:
    def __init__(self, text):
        self._text = text
        self.last_model = None
        self.last_contents = None

    def generate_content(self, model, contents):
        self.last_model = model
        self.last_contents = contents

        class _Resp:
            text = ""

        resp = _Resp()
        resp.text = self._text
        return resp


class _FakeGeminiClient:
    def __init__(self, text):
        class _Models:
            pass

        self.models = _Models()
        self.models.generate_content = _FakeGeminiModels(text).generate_content


class TestQwenUtils(unittest.TestCase):
    def test_query_qwen_uses_chat_template(self):
        tok = _TokenizerWithTemplate()
        gen = _Generator([{"generated_text": "hello"}])

        with patch("scripts.llm_utils._load_qwen_pipeline", return_value=(tok, gen)):
            out = llm_utils.query_qwen("Qwen/Qwen2.5-7B-Instruct", "hi there")

        self.assertEqual(out, "hello")
        self.assertEqual(gen.last_args[0], "TEMPLATE::hi there")
        self.assertEqual(tok.last_messages, [{"role": "user", "content": "hi there"}])
        self.assertFalse(tok.last_tokenize)
        self.assertTrue(tok.last_add_generation_prompt)

    def test_query_qwen_fallback_prompt_without_chat_template(self):
        tok = _TokenizerNoTemplate()
        gen = _Generator([{"generated_text": "ok"}])

        with patch("scripts.llm_utils._load_qwen_pipeline", return_value=(tok, gen)):
            out = llm_utils.query_qwen("Qwen/Qwen2.5-7B-Instruct", "ping")

        self.assertEqual(out, "ok")
        self.assertEqual(gen.last_args[0], "User: ping\nAssistant:")

    def test_query_qwen_batch_returns_outputs_in_order(self):
        tok = _TokenizerWithTemplate()
        gen = _Generator(
            [
                [{"generated_text": "r1"}],
                {"generated_text": "r2"},
            ]
        )

        with patch("scripts.llm_utils._load_qwen_pipeline", return_value=(tok, gen)):
            out = llm_utils.query_qwen_batch(
                "Qwen/Qwen2.5-7B-Instruct",
                ["alpha", "beta"],
            )

        self.assertEqual(out, ["r1", "r2"])
        self.assertEqual(gen.last_args[0], ["TEMPLATE::alpha", "TEMPLATE::beta"])

    def test_query_qwen_batch_raises_on_mismatched_output_count(self):
        tok = _TokenizerWithTemplate()
        gen = _Generator([[{"generated_text": "only-one"}]])

        with patch("scripts.llm_utils._load_qwen_pipeline", return_value=(tok, gen)):
            with self.assertRaises(RuntimeError):
                llm_utils.query_qwen_batch(
                    "Qwen/Qwen2.5-7B-Instruct",
                    ["a", "b"],
                )

    def test_validation(self):
        with self.assertRaises(ValueError):
            llm_utils.query_qwen("", "x")
        with self.assertRaises(ValueError):
            llm_utils.query_qwen("Qwen/Qwen2.5-7B-Instruct", "")
        with self.assertRaises(ValueError):
            llm_utils.query_qwen_batch("Qwen/Qwen2.5-7B-Instruct", [])
        with self.assertRaises(ValueError):
            llm_utils.query_qwen_batch("Qwen/Qwen2.5-7B-Instruct", ["ok", "   "])

    def test_query_gpt_mocked(self):
        client = _FakeOpenAIClient("gpt answer")
        with patch("scripts.llm_utils._get_openai_client", return_value=client):
            out = llm_utils.query_gpt("gpt-4.1-mini", "hello")
        self.assertEqual(out, "gpt answer")

    def test_query_gemini_mocked(self):
        client = _FakeGeminiClient("gemini answer")
        with patch("scripts.llm_utils._get_gemini_client", return_value=client):
            out = llm_utils.query_gemini("gemini-2.0-flash", "hello")
        self.assertEqual(out, "gemini answer")

    def test_closed_source_validation(self):
        with self.assertRaises(ValueError):
            llm_utils.query_gpt("", "x")
        with self.assertRaises(ValueError):
            llm_utils.query_gpt("gpt-4.1-mini", "")
        with self.assertRaises(ValueError):
            llm_utils.query_gemini("", "x")
        with self.assertRaises(ValueError):
            llm_utils.query_gemini("gemini-2.0-flash", "")


if __name__ == "__main__":
    unittest.main()
