"""Generic OpenAI-compatible provider adapter (DeepSeek, OpenRouter, Groq, DashScope, Ollama, vLLM, etc.)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from src.providers.base import ProviderSettings
from src.providers.openai import OpenAICompatTranscribeProvider, OpenAITranslateProvider


class OpenAICompatTranslateProvider(OpenAITranslateProvider):
    """OpenAI-compatible `/v1/chat/completions` translator for third-party and local LLM endpoints."""

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        post_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        effective_mode = settings.api_mode or "chat_completions"
        super().__init__(replace(settings, api_mode=effective_mode), post_fn=post_fn)


__all__ = [
    "OpenAICompatTranscribeProvider",
    "OpenAICompatTranslateProvider",
]
