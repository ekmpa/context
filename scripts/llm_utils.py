from __future__ import annotations

import os
from threading import Lock
from typing import Any

_PIPELINE_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_CACHE_LOCK = Lock()
_CLIENT_CACHE: dict[tuple[str, str], Any] = {}


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _resolve_qwen_device() -> str:
    _load_dotenv_if_available()
    configured_device = os.getenv("CONTEXT_QWEN_DEVICE", "").strip().lower()
    if configured_device in {"", "auto"}:
        configured_device = ""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for local Qwen generation") from exc

    if configured_device:
        if configured_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CONTEXT_QWEN_DEVICE requests CUDA, but torch.cuda.is_available() is False. "
                "Request a GPU or set CONTEXT_QWEN_DEVICE=cpu."
            )
        return configured_device

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_qwen_pipeline(model_name: str):
    """Load and cache tokenizer + causal LM for a model name."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for Qwen utilities. Install with: "
            "uv pip install transformers accelerate torch"
        ) from exc

    device = _resolve_qwen_device()
    cache_key = (model_name, device)

    with _CACHE_LOCK:
        cached = _PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Decoder-only models (for example Qwen) should use left padding for generation.
        tokenizer.padding_side = "left"
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        _PIPELINE_CACHE[cache_key] = (tokenizer, model)
        return tokenizer, model


def _get_model_device(model: Any):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Loaded Qwen model has no parameters to infer device placement") from exc


def _batch_generate(tokenizer: Any, model: Any, prompts: list[str], max_new_tokens: int = 512) -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for local Qwen generation") from exc

    if not prompts:
        return []

    device = _get_model_device(model)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    enc = {key: value.to(device) for key, value in enc.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        )

    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        prompt_lengths = [enc["input_ids"].shape[1]] * outputs.shape[0]
    else:
        prompt_lengths = attention_mask.sum(dim=1).tolist()

    completions: list[str] = []
    for row_index, prompt_len in enumerate(prompt_lengths):
        generated_ids = outputs[row_index, int(prompt_len):]
        text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        completions.append(text)
    return completions


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

    tokenizer, model = _load_qwen_pipeline(model_name.strip())
    prompt = _build_prompt(tokenizer, query)
    outputs = _batch_generate(tokenizer, model, [prompt], max_new_tokens=512)
    return outputs[0] if outputs else ""


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

    tokenizer, model = _load_qwen_pipeline(model_name.strip())
    prompts = [_build_prompt(tokenizer, query) for query in normalized_queries]
    outputs = _batch_generate(tokenizer, model, prompts, max_new_tokens=512)

    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"Unexpected output count from generator: got {len(outputs)}, expected {len(prompts)}"
        )

    return outputs


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
