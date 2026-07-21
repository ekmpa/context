import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import llm_utils
from scripts.context_core import llm as context_llm


class _TokenizerWithTemplate:
    eos_token_id = 7

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.last_messages = messages
        self.last_tokenize = tokenize
        self.last_add_generation_prompt = add_generation_prompt
        return f"TEMPLATE::{messages[0]['content']}"


class _TokenizerNoTemplate:
    eos_token_id = 9

    def __init__(self):
        self.pad_token_id = None
        self.padding_side = "right"


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


class _FakeModel:
    def __init__(self):
        self.to_calls = []
        self.device = None

    def to(self, device):
        self.to_calls.append(device)
        self.device = device
        return self


class TestQwenUtils(unittest.TestCase):
    def tearDown(self):
        llm_utils._PIPELINE_CACHE.clear()

    def test_load_qwen_pipeline_uses_tgm_style_device_setup(self):
        fake_model = _FakeModel()
        fake_tokenizer = _TokenizerNoTemplate()
        fake_transformers = SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda model_name: fake_model),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda model_name: fake_tokenizer),
        )
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

        with patch.dict("sys.modules", {"transformers": fake_transformers, "torch": fake_torch}):
            tokenizer, model = llm_utils._load_qwen_pipeline("Qwen/Qwen2.5-7B-Instruct")

        self.assertIs(tokenizer, fake_tokenizer)
        self.assertIs(model, fake_model)
        self.assertEqual(fake_model.to_calls, ["cpu"])
        self.assertEqual(fake_tokenizer.padding_side, "left")
        self.assertEqual(fake_tokenizer.pad_token_id, fake_tokenizer.eos_token_id)

    def test_query_qwen_uses_chat_template(self):
        tok = _TokenizerWithTemplate()
        model = object()

        with patch("scripts.llm_utils._load_qwen_pipeline", return_value=(tok, model)), patch(
            "scripts.llm_utils._batch_generate", return_value=["hello"]
        ) as mock_batch:
            out = llm_utils.query_qwen("Qwen/Qwen2.5-7B-Instruct", "hi there")

        self.assertEqual(out, "hello")
        self.assertEqual(mock_batch.call_args.args[2], ["TEMPLATE::hi there"])
        self.assertEqual(tok.last_messages, [{"role": "user", "content": "hi there"}])
        self.assertFalse(tok.last_tokenize)
        self.assertTrue(tok.last_add_generation_prompt)

    def test_query_qwen_fallback_prompt_without_chat_template(self):
        tok = _TokenizerNoTemplate()
        model = object()

        with patch("scripts.llm_utils._load_qwen_pipeline", return_value=(tok, model)), patch(
            "scripts.llm_utils._batch_generate", return_value=["ok"]
        ) as mock_batch:
            out = llm_utils.query_qwen("Qwen/Qwen2.5-7B-Instruct", "ping")

        self.assertEqual(out, "ok")
        self.assertEqual(mock_batch.call_args.args[2], ["User: ping\nAssistant:"])

    def test_query_qwen_batch_returns_outputs_in_order(self):
        tok = _TokenizerWithTemplate()
        model = object()

        with patch("scripts.llm_utils._load_qwen_pipeline", return_value=(tok, model)), patch(
            "scripts.llm_utils._batch_generate", return_value=["r1", "r2"]
        ) as mock_batch:
            out = llm_utils.query_qwen_batch(
                "Qwen/Qwen2.5-7B-Instruct",
                ["alpha", "beta"],
            )

        self.assertEqual(out, ["r1", "r2"])
        self.assertEqual(mock_batch.call_args.args[2], ["TEMPLATE::alpha", "TEMPLATE::beta"])

    def test_query_qwen_batch_raises_on_mismatched_output_count(self):
        tok = _TokenizerWithTemplate()
        model = object()

        with patch("scripts.llm_utils._load_qwen_pipeline", return_value=(tok, model)), patch(
            "scripts.llm_utils._batch_generate", return_value=["only-one"]
        ):
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


class TestContextCoreLlmBackendInference(unittest.TestCase):
    def test_chat_text_infers_hf_local_for_qwen_model(self):
        with patch.dict("os.environ", {}, clear=True), patch(
            "scripts.context_core.llm.infer_backend_from_model",
            return_value="hf-local",
        ), patch(
            "scripts.context_core.llm.query_qwen",
            side_effect=lambda model_name, prompt: f"{model_name}::{prompt}",
        ):
            out = context_llm.chat_text(
                "ping",
                model="Qwen/Qwen2.5-7B-Instruct",
                system_role="system prompt",
            )

        self.assertEqual(out, "Qwen/Qwen2.5-7B-Instruct::SYSTEM: system prompt\nUSER: ping")

    def test_completion_text_prefers_explicit_backend_override(self):
        with patch.dict("os.environ", {"CONTEXT_LLM_BACKEND": "hf-local"}, clear=True), patch(
            "scripts.context_core.llm.query_qwen",
            side_effect=lambda model_name, prompt: f"hf::{model_name}::{prompt}",
        ):
            out = context_llm.completion_text(
                "prompt body",
                model="gpt-5-mini",
            )

        self.assertEqual(out, "hf::gpt-5-mini::prompt body")


if __name__ == "__main__":
    unittest.main()
