import os
import re
import time
import random
from typing import Iterable

import openai
from openai import OpenAI

from llm_utils import query_qwen, query_qwen_batch

_CLIENT = None
_CHAT_CAPABILITIES: dict[str, dict[str, bool]] = {}


def _backend() -> str:
    return os.getenv("CONTEXT_LLM_BACKEND", "openai").strip().lower()


def _is_hf_local_backend() -> bool:
    return _backend() in {"hf-local", "hf_local", "local"}


def _coerce_chat_histories(user_inputs):
    if isinstance(user_inputs, str):
        return [{"role": "user", "content": user_inputs}]
    if isinstance(user_inputs, list):
        if all(isinstance(x, str) for x in user_inputs):
            return [
                {"role": "user" if i % 2 == 0 else "assistant", "content": x}
                for i, x in enumerate(user_inputs)
            ]
        if all(isinstance(x, dict) for x in user_inputs):
            return user_inputs
    raise ValueError("Invalid input for LLM chat call")


def _is_invalid_model_error(exc: openai.OpenAIError) -> bool:
    return "invalid model" in str(exc).lower()


def _is_context_length_error(exc: openai.OpenAIError) -> bool:
    text = str(exc).lower()
    return "maximum context length" in text or "context length" in text


def _is_chat_only_completion_endpoint_error(exc: openai.OpenAIError) -> bool:
    text = str(exc).lower()
    return (
        "chat model" in text
        and "not supported in the v1/completions endpoint" in text
    )


def _is_unsupported_parameter_error(exc: openai.OpenAIError, param_name: str) -> bool:
    text = str(exc).lower()
    return "unsupported parameter" in text and f"'{param_name.lower()}'" in text


def _is_unsupported_temperature_value_error(exc: openai.OpenAIError) -> bool:
    text = str(exc).lower()
    return "unsupported value" in text and "'temperature'" in text


def _get_chat_capabilities(model: str) -> dict[str, bool]:
    caps = _CHAT_CAPABILITIES.get(model)
    if caps is None:
        caps = {
            "supports_temperature": True,
            "supports_max_tokens": True,
        }
        _CHAT_CAPABILITIES[model] = caps
    return caps


def _chat_completion_with_fallbacks(
    *,
    model: str,
    messages: list[dict],
    temperature: float,
    stop: Iterable[str] | None = None,
    max_tokens: int | None = None,
):
    client = _get_client()
    caps = _get_chat_capabilities(model)
    base_kwargs = {
        "model": model,
        "messages": messages,
        "stop": list(stop) if stop is not None else None,
    }

    attempts = []
    if max_tokens is None:
        temperature_modes = [True, False] if caps["supports_temperature"] else [False]
        attempts = [{"use_temperature": mode} for mode in temperature_modes]
    else:
        token_fields = ["max_tokens", "max_completion_tokens"]
        if not caps["supports_max_tokens"]:
            token_fields = ["max_completion_tokens", "max_tokens"]

        temperature_modes = [True, False] if caps["supports_temperature"] else [False]
        for token_field in token_fields:
            for mode in temperature_modes:
                attempts.append({"token_field": token_field, "use_temperature": mode})

    last_exc = None
    for attempt in attempts:
        kwargs = dict(base_kwargs)
        if attempt.get("use_temperature", False):
            kwargs["temperature"] = temperature
        token_field = attempt.get("token_field")
        if token_field is not None and max_tokens is not None:
            kwargs[token_field] = max_tokens

        try:
            response = client.chat.completions.create(**kwargs)
            if token_field == "max_tokens":
                caps["supports_max_tokens"] = True
            if attempt.get("use_temperature", False):
                caps["supports_temperature"] = True
            return response
        except openai.OpenAIError as exc:
            last_exc = exc
            if _is_unsupported_temperature_value_error(exc):
                caps["supports_temperature"] = False
                continue
            if token_field == "max_tokens" and _is_unsupported_parameter_error(exc, "max_tokens"):
                caps["supports_max_tokens"] = False
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM chat completion failed before any attempt")


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


def _compute_retry_wait_seconds(
    *,
    attempt: int,
    base_wait: float,
    message: str,
    consecutive_rate_limits: int,
) -> float:
    # Apply exponential backoff with jitter, and cool down more aggressively
    # when a burst of consecutive rate-limit errors is detected.
    retry_wait = max(base_wait, base_wait * (2 ** max(0, attempt - 1)))
    lowered = message.lower()
    if "rate limit" in lowered:
        parsed_wait = _parse_retry_after_seconds(message)
        if parsed_wait is not None:
            retry_wait = max(retry_wait, parsed_wait + 0.2)
        if consecutive_rate_limits > 1:
            retry_wait = max(retry_wait, base_wait * (2 ** min(consecutive_rate_limits, 6)))

    jitter = random.uniform(0.0, max(0.05, retry_wait * 0.15))
    return min(retry_wait + jitter, 30.0)


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
        chat_histories = _coerce_chat_histories(user_inputs)

        merged = [{"role": "system", "content": system_role}] + chat_histories
        prompt = _messages_to_text(merged)
        return query_qwen(model.strip(), prompt)

    consecutive_rate_limits = 0
    for attempt in range(1, num_retries + 1):
        try:
            chat_histories = _coerce_chat_histories(user_inputs)

            messages = [{"role": "system", "content": system_role}] + chat_histories
            response = _chat_completion_with_fallbacks(
                model=model,
                messages=messages,
                temperature=temperature,
                stop=None,
                max_tokens=None,
            )
            return "".join(choice.message.content or "" for choice in response.choices)
        except openai.OpenAIError as exc:
            if _is_invalid_model_error(exc):
                raise RuntimeError(
                    "OpenAI backend rejected the model id. "
                    "If using Hugging Face models like Qwen/*, either rely on backend inference "
                    "from the model name or set CONTEXT_LLM_BACKEND=hf-local explicitly."
                ) from exc
            if _is_context_length_error(exc):
                raise RuntimeError("LLM prompt exceeds model context window") from exc

            message = str(exc)
            if "rate limit" in message.lower():
                consecutive_rate_limits += 1
            else:
                consecutive_rate_limits = 0

            if attempt == num_retries:
                break

            retry_wait = _compute_retry_wait_seconds(
                attempt=attempt,
                base_wait=waiting,
                message=message,
                consecutive_rate_limits=consecutive_rate_limits,
            )
            print(
                f"{exc}. Retrying in {retry_wait:.2f}s "
                f"(attempt {attempt + 1}/{num_retries})..."
            )
            time.sleep(retry_wait)
    raise RuntimeError("LLM chat call failed after retries")


def chat_text_batch(
    user_inputs_batch,
    *,
    model: str,
    system_role: str,
    temperature: float = 1.0,
    num_retries: int = 3,
    waiting: float = 1.0,
) -> list[str]:
    if not isinstance(user_inputs_batch, list) or not user_inputs_batch:
        raise ValueError("user_inputs_batch must be a non-empty list")

    if _is_hf_local_backend():
        prompts: list[str] = []
        for user_inputs in user_inputs_batch:
            chat_histories = _coerce_chat_histories(user_inputs)
            merged = [{"role": "system", "content": system_role}] + chat_histories
            prompts.append(_messages_to_text(merged))
        return query_qwen_batch(model.strip(), prompts)

    # Keep remote APIs simple and stable; batch locally only for hf-local.
    return [
        chat_text(
            user_inputs,
            model=model,
            system_role=system_role,
            temperature=temperature,
            num_retries=num_retries,
            waiting=waiting,
        )
        for user_inputs in user_inputs_batch
    ]


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

    consecutive_rate_limits = 0
    for attempt in range(1, num_retries + 1):
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
            if _is_chat_only_completion_endpoint_error(exc):
                # Some OpenAI models (for example gpt-4.1-mini) are chat-only.
                # If a caller asks completion_text() with one of these models,
                # fall back to the chat endpoint using the prompt as a user turn.
                chat_response = _chat_completion_with_fallbacks(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stop=stop,
                    max_tokens=max_tokens,
                )

                return "".join(choice.message.content or "" for choice in chat_response.choices)

            if _is_invalid_model_error(exc):
                raise RuntimeError(
                    "OpenAI backend rejected the model id. "
                    "If using Hugging Face models like Qwen/*, either rely on backend inference "
                    "from the model name or set CONTEXT_LLM_BACKEND=hf-local explicitly."
                ) from exc
            if _is_context_length_error(exc):
                raise RuntimeError("LLM prompt exceeds model context window") from exc

            message = str(exc)
            if "rate limit" in message.lower():
                consecutive_rate_limits += 1
            else:
                consecutive_rate_limits = 0

            if attempt == num_retries:
                break

            retry_wait = _compute_retry_wait_seconds(
                attempt=attempt,
                base_wait=waiting,
                message=message,
                consecutive_rate_limits=consecutive_rate_limits,
            )

            print(
                f"{exc}. Retrying in {retry_wait:.2f}s "
                f"(attempt {attempt + 1}/{num_retries})..."
            )
            time.sleep(retry_wait)
    raise RuntimeError("LLM completion call failed after retries")


def completion_text_batch(
    prompts: list[str],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    stop: Iterable[str] | None = None,
    num_retries: int = 5,
    waiting: float = 1.0,
    logit_bias: dict[str, int] | None = None,
) -> list[str]:
    if not prompts:
        return []

    if _is_hf_local_backend():
        outputs = query_qwen_batch(model.strip(), prompts)
        if stop:
            trimmed: list[str] = []
            for text in outputs:
                cutoff = None
                for marker in stop:
                    idx = text.find(marker)
                    if idx >= 0:
                        cutoff = idx if cutoff is None else min(cutoff, idx)
                if cutoff is not None:
                    text = text[:cutoff]
                trimmed.append(text)
            return trimmed
        return outputs

    return [
        completion_text(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            num_retries=num_retries,
            waiting=waiting,
            logit_bias=logit_bias,
        )
        for prompt in prompts
    ]
