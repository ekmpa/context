from __future__ import annotations

import os
from threading import Lock
from typing import Any

_PIPELINE_CACHE: dict[str, tuple[Any, Any]] = {}
_CACHE_LOCK = Lock()
_CLIENT_CACHE: dict[tuple[str, str], Any] = {}


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _load_qwen_pipeline(model_name: str):
    """Load and cache tokenizer + generation pipeline for a model name."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for Qwen utilities. Install with: "
            "uv pip install transformers accelerate torch"
        ) from exc

    with _CACHE_LOCK:
        cached = _PIPELINE_CACHE.get(model_name)
        if cached is not None:
            return cached

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        # Prevent transformers from mixing default max_length (often 20) with
        # explicit max_new_tokens, which triggers noisy warnings.
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "max_length"):
            generation_config.max_length = None

        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        )

        _PIPELINE_CACHE[model_name] = (tokenizer, generator)
        return tokenizer, generator


def _get_openai_client(api_key: str | None = None):
    _load_dotenv_if_available()
    resolved_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not resolved_key:
        raise ValueError("Missing OPENAI_API_KEY in environment or .env file")

    cache_key = ("openai", resolved_key)
    with _CACHE_LOCK:
        cached = _CLIENT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency for GPT utilities. Install with: uv pip install openai python-dotenv"
            ) from exc

        client = OpenAI(api_key=resolved_key)
        _CLIENT_CACHE[cache_key] = client
        return client


def _get_gemini_client(api_key: str | None = None):
    _load_dotenv_if_available()
    resolved_key = (
        api_key
        or os.environ.get("GOOGLE_API_KEY", "")
        or os.environ.get("GOOGLE_AIS_API_KEY", "")
    ).strip()
    if not resolved_key:
        raise ValueError("Missing GOOGLE_API_KEY (or GOOGLE_AIS_API_KEY) in environment or .env file")

    cache_key = ("gemini", resolved_key)
    with _CACHE_LOCK:
        cached = _CLIENT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency for Gemini utilities. Install with: uv pip install google-genai python-dotenv"
            ) from exc

        client = genai.Client(api_key=resolved_key)
        _CLIENT_CACHE[cache_key] = client
        return client


def _build_prompt(tokenizer: Any, query: str) -> str:
    messages = [{"role": "user", "content": query.strip()}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {query.strip()}\nAssistant:"


def _extract_generated_text(output_item: Any) -> str:
    # Transformers pipelines may return dict items or nested list items.
    if isinstance(output_item, list):
        if not output_item:
            return ""
        first = output_item[0]
        if isinstance(first, dict):
            return str(first.get("generated_text", "")).strip()
        return str(first).strip()
    if isinstance(output_item, dict):
        return str(output_item.get("generated_text", "")).strip()
    return str(output_item).strip()


def query_qwen(model_name: str, query: str) -> str:
    """Generate a single response from a Qwen-style chat model.

    Args:
        model_name: Hugging Face model id, for example ``Qwen/Qwen2.5-7B-Instruct``.
        query: User prompt.

    Returns:
        Generated assistant text.
    """
    if not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not query.strip():
        raise ValueError("query must be a non-empty string")

    tokenizer, generator = _load_qwen_pipeline(model_name.strip())
    prompt = _build_prompt(tokenizer, query)

    outputs = generator(
        prompt,
        max_new_tokens=512,
        max_length=None,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    first_output = outputs[0] if outputs else ""
    return _extract_generated_text(first_output)


def query_qwen_batch(model_name: str, queries: list[str]) -> list[str]:
    """Generate responses for a batch of prompts using one pipeline invocation.

    Args:
        model_name: Hugging Face model id, for example ``Qwen/Qwen2.5-7B-Instruct``.
        queries: List of prompts.

    Returns:
        Generated responses aligned with input order.
    """
    if not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not queries:
        raise ValueError("queries must contain at least one prompt")

    normalized_queries = [q.strip() for q in queries]
    if any(not q for q in normalized_queries):
        raise ValueError("queries must not contain empty prompts")

    tokenizer, generator = _load_qwen_pipeline(model_name.strip())
    prompts = [_build_prompt(tokenizer, query) for query in normalized_queries]

    outputs = generator(
        prompts,
        max_new_tokens=512,
        max_length=None,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"Unexpected output count from generator: got {len(outputs)}, expected {len(prompts)}"
        )

    return [_extract_generated_text(item) for item in outputs]


def query_gpt(model_name: str, query: str, *, api_key: str | None = None) -> str:
    """Query an OpenAI GPT model via the Responses API."""
    if not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not query.strip():
        raise ValueError("query must be a non-empty string")

    client = _get_openai_client(api_key=api_key)
    response = client.responses.create(model=model_name.strip(), input=query.strip())
    return str(getattr(response, "output_text", "")).strip()


def query_gemini(model_name: str, query: str, *, api_key: str | None = None) -> str:
    """Query a Gemini model via the Google GenAI SDK."""
    if not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not query.strip():
        raise ValueError("query must be a non-empty string")

    client = _get_gemini_client(api_key=api_key)
    response = client.models.generate_content(model=model_name.strip(), contents=query.strip())
    text = getattr(response, "text", "")
    return str(text).strip() if text is not None else ""
