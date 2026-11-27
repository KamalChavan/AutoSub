"""Abstract interfaces and shared data types for multi-provider transcription and translation."""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.http_client import ProviderApiError


@dataclass(frozen=True)
class ProviderSettings:
    """Resolved runtime connection settings for a named provider instance."""

    name: str
    protocol: str
    api_key: str
    api_base: str
    proxy: str = ""
    timeout_seconds: int = 180
    api_mode: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    """Normalized provider + model entry for transcription or translation chains."""

    provider: str
    model: str
    service_tier: str = "standard"
    reasoning_effort: str = ""
    thinking_level: str = ""
    temperature: float | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.provider, self.model, self.service_tier)

    @property
    def display_label(self) -> str:
        prefix = f"{self.provider}/" if self.provider else ""
        detail = self.reasoning_effort or self.service_tier
        return f"{prefix}{self.model}({detail})" if detail else f"{prefix}{self.model}"


@dataclass
class TranscribedWord:
    """Single word with chunk-local start/end timestamps (seconds)."""

    start: float
    end: float
    text: str
    space_before: bool = True
    speaker_id: str | None = None


@dataclass
class TranscribeChunkResult:
    """Normalized output from transcribing one audio chunk."""

    words: list[TranscribedWord]
    usage_summary: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslateBatchResult:
    """Normalized structured JSON output and token usage from translating one batch."""

    parsed: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    service_tier: str = "standard"


def _extract_balanced_json_object(text: str) -> str | None:
    """Extract the first balanced top-level `{...}` JSON object span, respecting string quotes."""
    start = -1
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : idx + 1]
    return None


def _repair_json_syntax(text: str) -> str:
    """Safely fix common LLM JSON glitches (trailing commas & raw newlines in strings) without touching valid content."""
    out: list[str] = []
    in_string = False
    escape = False
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if in_string:
            if escape:
                out.append(ch)
                escape = False
            elif ch == "\\":
                out.append(ch)
                escape = True
            elif ch == '"':
                out.append(ch)
                in_string = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            # Drop comma if the next non-whitespace token is a closing `]` or `}`
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "]}":
                i += 1
                continue

        out.append(ch)
        i += 1
    return "".join(out)


def parse_json_response_text(raw_text: str, *, provider_name: str = "API") -> dict[str, Any]:
    """Strip markdown fences, safely repair common LLM JSON glitches, and return a JSON dict."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates: list[str] = [cleaned]
    extracted = _extract_balanced_json_object(cleaned)
    if extracted and extracted != cleaned:
        candidates.append(extracted)

    last_exc: Exception | None = None
    for cand in candidates:
        for variant in (cand, _repair_json_syntax(cand)):
            try:
                parsed = json.loads(variant)
            except json.JSONDecodeError as exc:
                last_exc = exc
                continue
            if isinstance(parsed, dict):
                return parsed
            last_exc = None

    if last_exc is not None:
        raise ProviderApiError(
            "invalid_json",
            f"返回非合法 JSON: {last_exc}",
            http_status=503,
            provider=provider_name,
        ) from last_exc
    raise ProviderApiError(
        "invalid_json",
        "模型未返回 JSON 对象",
        http_status=503,
        provider=provider_name,
    )


class TranscribeProvider(ABC):
    """Speech-to-text provider interface returning native word-level timestamps."""

    max_chunk_bytes: int | None = None

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings

    @abstractmethod
    def transcribe_chunk(
        self,
        audio_path: Path,
        *,
        model: str,
        options: dict[str, Any] | None = None,
        max_retries: int = 6,
        sleep_fn: Callable[[float], None] = time.sleep,
        progress_fn: Callable[[str], None] | None = None,
    ) -> TranscribeChunkResult:
        """Transcribe a single chunk audio file into chunk-local timed words."""


class TranslateProvider(ABC):
    """Structured-JSON LLM provider interface for subtitle batch translation."""

    def __init__(self, settings: ProviderSettings) -> None:
        self.settings = settings

    @abstractmethod
    def generate_json(
        self,
        spec: ModelSpec,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> TranslateBatchResult:
        """Execute a single translation request and return parsed JSON + normalized token usage."""
