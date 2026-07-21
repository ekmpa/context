from __future__ import annotations

MODEL_BACKEND_PREFIX_MAP: dict[str, tuple[str, ...]] = {
    "hf-local": ("qwen", "llama"),
    "openai": ("gpt", "o1", "o3", "o4", "text-"),
}


def infer_backend_from_model(model_name: str) -> str:
    lower_name = (model_name or "").strip().lower()
    for backend, prefixes in MODEL_BACKEND_PREFIX_MAP.items():
        if any(lower_name.startswith(prefix) for prefix in prefixes):
            return backend
    return "openai"


def is_hf_local_backend(backend: str | None = None) -> bool:
    resolved = (backend or "").strip().lower()
    if not resolved:
        return False
    return resolved in {"hf-local", "hf_local", "local"}
