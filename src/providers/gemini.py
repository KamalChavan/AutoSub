"""Google Gemini provider adapters for transcription (Files API + Interactions) and translation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from src.http_client import ProviderApiError, delete_url, post_json, upload_resumable_file
from src.providers.base import (
    ModelSpec,
    ProviderSettings,
    TranscribeChunkResult,
    TranscribedWord,
    TranscribeProvider,
    TranslateBatchResult,
    TranslateProvider,
    parse_json_response_text,
)

DEFAULT_GEMINI_TRANSCRIBE_MODEL = "gemini-3.5-transcribe"


def call_gemini_interactions_transcribe(
    file_uri: str,
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_TRANSCRIBE_MODEL,
    api_base: str = "https://generativelanguage.googleapis.com/v1beta",
    proxy: str = "",
    max_retries: int = 6,
    sleep_fn: Callable[[float], None] = time.sleep,
    post_fn: Callable[..., dict[str, Any]] | None = None,
    progress_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Call Gemini `/v1beta/interactions` with verbatim mode + word-level timestamps."""
    model_id = model.removeprefix("google/").removeprefix("models/")
    payload = {
        "model": model_id,
        "input": [{"type": "audio", "uri": file_uri, "mime_type": "audio/mp3"}],
        "generation_config": {
            "transcription_config": {
                "mode": {
                    "type": "verbatim",
                    "timestamp_granularities": ["word"],
                }
            }
        },
    }
    url = f"{api_base.rstrip('/')}/interactions"
    requester = post_fn or post_json
    last_err: Exception | None = None

    for attempt in range(1, max(1, max_retries) + 1):
        try:
            return requester(url, payload, api_key=api_key, proxy=proxy, timeout=360)
        except ProviderApiError as api_err:
            last_err = api_err
            if api_err.is_429_rate_limit and api_err.retry_after_seconds is not None and attempt < max_retries:
                wait_s = api_err.retry_after_seconds + 3
                if progress_fn:
                    progress_fn(f"    [转写限流 {attempt}/{max_retries}] 触发 429，等待 {wait_s}s 后重试...")
                sleep_fn(wait_s)
                continue
            if api_err.is_503_transient and attempt < max_retries:
                wait_s = min(30, 8 * attempt)
                if progress_fn:
                    progress_fn(f"    [转写高负载 {attempt}/{max_retries}] 遇到 {api_err.code}，{wait_s}s 后重试...")
                sleep_fn(wait_s)
                continue
            raise
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                wait_s = min(30, 8 * attempt)
                if progress_fn:
                    progress_fn(f"    [转写重试 {attempt}/{max_retries}] 网络/异常 ({type(exc).__name__})，{wait_s}s 后重试...")
                sleep_fn(wait_s)
                continue
            raise
    raise RuntimeError(f"{model_id} 调用失败: {last_err}")


def extract_gemini_words(resp: dict[str, Any]) -> list[TranscribedWord]:
    """Extract chunk-local word timestamps from a Gemini `/v1beta/interactions` response."""
    words: list[TranscribedWord] = []
    for step in resp.get("steps") or []:
        for content in step.get("content") or []:
            for ann in content.get("annotations") or []:
                if ann.get("type") != "word_info":
                    continue
                txt = str(ann.get("text") or "").strip()
                if not txt:
                    continue
                ws = float(str(ann.get("start_offset", "0s")).rstrip("s"))
                we = float(str(ann.get("end_offset", "0s")).rstrip("s"))
                words.append(TranscribedWord(start=ws, end=we, text=txt, space_before=True))
    return words


class GeminiTranscribeProvider(TranscribeProvider):
    """Gemini Files API + `/v1beta/interactions` speech-to-text adapter."""

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        post_fn: Callable[..., dict[str, Any]] | None = None,
        upload_fn: Callable[..., dict[str, Any]] | None = None,
        delete_fn: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(settings)
        self.post_fn = post_fn or post_json
        self.upload_fn = upload_fn or upload_resumable_file
        self.delete_fn = delete_fn or delete_url

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
        api_base = self.settings.api_base or "https://generativelanguage.googleapis.com/v1beta"
        finfo = self.upload_fn(
            audio_path,
            mime_type="audio/mp3",
            api_key=self.settings.api_key,
            api_base=api_base,
            proxy=self.settings.proxy,
            timeout=300,
        )
        try:
            resp = call_gemini_interactions_transcribe(
                str(finfo["uri"]),
                api_key=self.settings.api_key,
                model=model,
                api_base=api_base,
                proxy=self.settings.proxy,
                max_retries=max_retries,
                sleep_fn=sleep_fn,
                post_fn=self.post_fn,
                progress_fn=progress_fn,
            )
            words = extract_gemini_words(resp)
            usage = resp.get("usage") or {}
            summary = f"Token: {usage.get('total_input_tokens')} in / {usage.get('total_output_tokens')} out"
            return TranscribeChunkResult(words=words, usage_summary=summary, raw_response=resp)
        finally:
            file_name = str(finfo.get("name") or "")
            if file_name:
                self.delete_fn(
                    f"{api_base.rstrip('/')}/{file_name}",
                    api_key=self.settings.api_key,
                    proxy=self.settings.proxy,
                    timeout=30,
                )


class GeminiTranslateProvider(TranslateProvider):
    """Gemini `:generateContent` structured-JSON subtitle translator."""

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        post_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(settings)
        self.post_fn = post_fn or post_json

    def generate_json(
        self,
        spec: ModelSpec,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> TranslateBatchResult:
        model = spec.model.removeprefix("google/").removeprefix("models/")
        gen_cfg: dict[str, Any] = {
            "temperature": 0.2 if spec.temperature is None else spec.temperature,
            "responseMimeType": "application/json",
        }
        if "lite" not in model.lower():
            level = (spec.thinking_level or "LOW").upper()
            gen_cfg["thinkingConfig"] = {"thinkingLevel": level}

        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_cfg,
        }
        if spec.service_tier and spec.service_tier != "standard":
            payload["service_tier"] = spec.service_tier

        api_base = (self.settings.api_base or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{api_base}/models/{model}:generateContent"
        resp = self.post_fn(
            url,
            payload,
            api_key=self.settings.api_key,
            proxy=self.settings.proxy,
            timeout=self.settings.timeout_seconds,
        )
        candidates = resp.get("candidates") or []
        if not candidates:
            raise ProviderApiError("empty_candidates", "未返回有效候选结果", http_status=503, provider="Gemini")
        parts = candidates[0].get("content", {}).get("parts") or []
        raw_text = "".join(str(p.get("text", "")) for p in parts).strip()
        parsed = parse_json_response_text(raw_text, provider_name="Gemini")
        usage = resp.get("usageMetadata") or {}
        return TranslateBatchResult(
            parsed=parsed,
            input_tokens=int(usage.get("promptTokenCount") or 0),
            output_tokens=int(usage.get("candidatesTokenCount") or 0),
            cached_tokens=int(usage.get("cachedContentTokenCount") or 0),
            service_tier=str(usage.get("serviceTier") or spec.service_tier or "standard"),
        )
