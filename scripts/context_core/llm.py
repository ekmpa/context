import os
import time
from typing import Iterable

import openai
from openai import OpenAI

_CLIENT = None


def _backend() -> str:
    return os.getenv("CONTEXT_LLM_BACKEND", "openai").strip().lower()


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
            print(f"{exc}. Retrying...")
            time.sleep(waiting)
    raise RuntimeError("LLM completion call failed after retries")
