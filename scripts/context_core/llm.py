import os
import re
import time
import importlib
from typing import Iterable

import openai
from openai import OpenAI

_CLIENT = None


def _backend() -> str:
    return os.getenv("CONTEXT_LLM_BACKEND", "openai").strip().lower()


def _is_hf_local_backend() -> bool:
    return _backend() in {"hf-local", "hf_local", "local"}


def _load_qwen_query_fn():
    for module_name in ("scripts.llm_utils", "llm_utils"):
        try:
            module = importlib.import_module(module_name)
            query_qwen = getattr(module, "query_qwen", None)
            if callable(query_qwen):
                return query_qwen
        except Exception:
            continue
    raise RuntimeError(
        "CONTEXT_LLM_BACKEND=hf-local requires llm_utils.query_qwen to be importable"
    )


def _is_invalid_model_error(exc: openai.OpenAIError) -> bool:
    return "invalid model" in str(exc).lower()


def _is_context_length_error(exc: openai.OpenAIError) -> bool:
    text = str(exc).lower()
    return "maximum context length" in text or "context length" in text


def _parse_retry_after_seconds(message: str) -> float | None:
    # OpenAI errors often include: "Please try again in 1.824s" or "73ms".
    match = re.search(r"try again in\s+([0-9]*\.?[0-9]+)\s*(ms|s)", message, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "ms":
        return max(0.0, value / 1000.0)
    return max(0.0, value)


def _messages_to_text(messages: list[dict]) -> str:
    lines = []
    for item in messages:
        role = str(item.get("role", "user")).strip().upper()
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _get_client() -> OpenAI:
    global _CLIENT
    if _backend() != "openai":
        raise RuntimeError(f"Unsupported CONTEXT_LLM_BACKEND={_backend()}")
    if _CLIENT is None:
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY is required")
        _CLIENT = OpenAI()
    return _CLIENT


def chat_text(
    user_inputs,
    *,
    model: str,
    system_role: str,
    temperature: float = 1.0,
    num_retries: int = 3,
    waiting: float = 1.0,
) -> str:
    if _is_hf_local_backend():
        if isinstance(user_inputs, str):
            chat_histories = [{"role": "user", "content": user_inputs}]
        elif isinstance(user_inputs, list):
            if all(isinstance(x, str) for x in user_inputs):
                chat_histories = [
                    {"role": "user" if i % 2 == 0 else "assistant", "content": x}
                    for i, x in enumerate(user_inputs)
                ]
            elif all(isinstance(x, dict) for x in user_inputs):
                chat_histories = user_inputs
            else:
                raise ValueError("Invalid input for LLM chat call")
        else:
            raise ValueError("Invalid input for LLM chat call")

        merged = [{"role": "system", "content": system_role}] + chat_histories
        prompt = _messages_to_text(merged)
        query_qwen = _load_qwen_query_fn()
        return query_qwen(model.strip(), prompt)

    for _ in range(num_retries):
        try:
            if isinstance(user_inputs, str):
                chat_histories = [{"role": "user", "content": user_inputs}]
            elif isinstance(user_inputs, list):
                if all(isinstance(x, str) for x in user_inputs):
                    chat_histories = [
                        {"role": "user" if i % 2 == 0 else "assistant", "content": x}
                        for i, x in enumerate(user_inputs)
                    ]
                elif all(isinstance(x, dict) for x in user_inputs):
                    chat_histories = user_inputs
                else:
                    raise ValueError("Invalid input for LLM chat call")
            else:
                raise ValueError("Invalid input for LLM chat call")

            messages = [{"role": "system", "content": system_role}] + chat_histories
            response = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return "".join(choice.message.content or "" for choice in response.choices)
        except openai.OpenAIError as exc:
            if _is_invalid_model_error(exc):
                raise RuntimeError(
                    "OpenAI backend rejected the model id. "
                    "If using Hugging Face models like Qwen/*, set CONTEXT_LLM_BACKEND=hf-local."
                ) from exc
            print(f"{exc}. Retrying...")
            time.sleep(waiting)
    raise RuntimeError("LLM chat call failed after retries")


def completion_text(
    prompt: str,
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    stop: Iterable[str] | None = None,
    num_retries: int = 5,
    waiting: float = 1.0,
    logit_bias: dict[str, int] | None = None,
) -> str:
    if _is_hf_local_backend():
        query_qwen = _load_qwen_query_fn()
        text = query_qwen(model.strip(), prompt)
        if stop:
            cutoff = None
            for marker in stop:
                idx = text.find(marker)
                if idx >= 0:
                    cutoff = idx if cutoff is None else min(cutoff, idx)
            if cutoff is not None:
                text = text[:cutoff]
        return text

    for _ in range(num_retries):
        try:
            response = _get_client().completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=list(stop) if stop is not None else None,
                logit_bias=logit_bias,
            )
            return response.choices[0].text or ""
        except openai.OpenAIError as exc:
            if _is_invalid_model_error(exc):
                raise RuntimeError(
                    "OpenAI backend rejected the model id. "
                    "If using Hugging Face models like Qwen/*, set CONTEXT_LLM_BACKEND=hf-local."
                ) from exc
            if _is_context_length_error(exc):
                raise RuntimeError("LLM prompt exceeds model context window") from exc

            message = str(exc)
            retry_wait = waiting
            if "rate limit" in message.lower():
                parsed_wait = _parse_retry_after_seconds(message)
                if parsed_wait is not None:
                    # Add a tiny buffer to avoid retrying before the bucket resets.
                    retry_wait = max(waiting, parsed_wait + 0.05)

            print(f"{exc}. Retrying...")
            time.sleep(retry_wait)
    raise RuntimeError("LLM completion call failed after retries")
