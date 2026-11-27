"""OpenAI provider adapters: Responses API & Chat Completions translation + timestamped Whisper-compatible STT."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from src.http_client import ProviderApiError, post_json, post_multipart
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

_UNTIMESTAMPED_STT_MODELS = (
    "gpt-transcribe",
    "gpt-live-transcribe",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
)
_NO_NONE_REASONING_PREFIXES = (
    "gpt-6.2-sol",
    "gpt-6.2-astra",
    "gpt-6.1-sol",
    "gpt-6-astra",
)
_NO_FLEX_TIER_PREFIXES = (
    "gpt-6.2-sol",
    "gpt-6.2-astra",
)


def _bare_model_id(model: str) -> str:
    cleaned = model.strip().lower()
    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1]
    return cleaned


def is_openai_reasoning_model(model: str) -> bool:
    """Return True if the model belongs to an OpenAI reasoning family (GPT-6 / GPT-5 / o-series)."""
    bare = _bare_model_id(model)
    return bare.startswith(("gpt-6", "gpt-5", "o1", "o3", "o4"))


def _resolve_reasoning_effort(model: str, configured_effort: str) -> str:
    bare = _bare_model_id(model)
    effort = (configured_effort or "").strip().lower()
    if effort:
        if effort == "none" and any(bare.startswith(p) for p in _NO_NONE_REASONING_PREFIXES):
            raise ValueError(f"{model} 不支持 reasoning_effort='none'，请改用 low / medium / high")
        return effort
    if is_openai_reasoning_model(model):
        return "low"
    return ""


def extract_responses_text(resp: dict[str, Any], *, provider_name: str = "OpenAI") -> str:
    """Extract assistant output text from an OpenAI `/v1/responses` JSON payload."""
    direct = resp.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    chunks: list[str] = []
    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in (None, "message"):
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "refusal":
                raise ProviderApiError(
                    "refusal",
                    str(part.get("refusal") or "模型拒绝输出"),
                    http_status=400,
                    provider=provider_name,
                )
            if ptype in ("output_text", "text"):
                txt = part.get("text")
                if isinstance(txt, str):
                    chunks.append(txt)
    return "".join(chunks).strip()


def extract_chat_completions_text(resp: dict[str, Any], *, provider_name: str = "OpenAI") -> str:
    """Extract assistant output text from a `/v1/chat/completions` JSON payload."""
    choices = resp.get("choices") or []
    if not choices:
        raise ProviderApiError("empty_choices", "未返回有效候选结果", http_status=503, provider=provider_name)
    msg = choices[0].get("message") or {}
    if msg.get("refusal"):
        raise ProviderApiError("refusal", str(msg["refusal"]), http_status=400, provider=provider_name)
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(p.get("text") or "") for p in content if isinstance(p, dict)]
        return "".join(parts).strip()
    return ""


class OpenAITranslateProvider(TranslateProvider):
    """OpenAI translator supporting `/v1/responses` (recommended for GPT-6.2) and `/v1/chat/completions`."""

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        post_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(settings)
        self.post_fn = post_fn or post_json

    def _auth_headers(self) -> dict[str, str]:
        headers = dict(self.settings.extra_headers)
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _generate_via_responses(
        self,
        spec: ModelSpec,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> TranslateBatchResult:
        api_base = (self.settings.api_base or "https://api.openai.com/v1").rstrip("/")
        url = f"{api_base}/responses"
        reasoning_effort = _resolve_reasoning_effort(spec.model, spec.reasoning_effort)
        bare = _bare_model_id(spec.model)

        # GPT-6.2 appends top-level `instructions` after `input`; placing reusable system_prompt as
        # the first `developer` message in `input` keeps the prefix cacheable across subtitle batches.
        payload: dict[str, Any] = {
            "model": spec.model,
            "input": [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {"format": {"type": "json_object"}},
        }
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        elif spec.temperature is not None:
            payload["temperature"] = spec.temperature
        elif not is_openai_reasoning_model(spec.model):
            payload["temperature"] = 0.2

        tier = (spec.service_tier or "").strip().lower()
        if tier and tier not in ("standard", "default", "auto"):
            if tier == "flex" and any(bare.startswith(p) for p in _NO_FLEX_TIER_PREFIXES):
                pass
            else:
                payload["service_tier"] = tier

        resp = self.post_fn(
            url,
            payload,
            headers=self._auth_headers(),
            proxy=self.settings.proxy,
            timeout=self.settings.timeout_seconds,
            provider_name=self.settings.name,
        )
        raw_text = extract_responses_text(resp, provider_name=self.settings.name)
        if not raw_text:
            raise ProviderApiError("empty_output", "未返回有效文本内容", http_status=503, provider=self.settings.name)
        parsed = parse_json_response_text(raw_text, provider_name=self.settings.name)
        usage = resp.get("usage") or {}
        in_details = usage.get("input_tokens_details") or {}
        return TranslateBatchResult(
            parsed=parsed,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cached_tokens=int(in_details.get("cached_tokens") or 0),
            service_tier=str(resp.get("service_tier") or spec.service_tier or "standard"),
        )

    def _generate_via_chat_completions(
        self,
        spec: ModelSpec,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> TranslateBatchResult:
        api_base = (self.settings.api_base or "https://api.openai.com/v1").rstrip("/")
        url = f"{api_base}/chat/completions"
        reasoning_effort = _resolve_reasoning_effort(spec.model, spec.reasoning_effort)
        bare = _bare_model_id(spec.model)

        payload: dict[str, Any] = {
            "model": spec.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        else:
            payload["temperature"] = 0.2 if spec.temperature is None else spec.temperature

        tier = (spec.service_tier or "").strip().lower()
        if tier and tier not in ("standard", "default", "auto"):
            if not (tier == "flex" and any(bare.startswith(p) for p in _NO_FLEX_TIER_PREFIXES)):
                payload["service_tier"] = tier

        resp = self.post_fn(
            url,
            payload,
            headers=self._auth_headers(),
            proxy=self.settings.proxy,
            timeout=self.settings.timeout_seconds,
            provider_name=self.settings.name,
        )
        raw_text = extract_chat_completions_text(resp, provider_name=self.settings.name)
        if not raw_text:
            raise ProviderApiError("empty_output", "未返回有效文本内容", http_status=503, provider=self.settings.name)
        parsed = parse_json_response_text(raw_text, provider_name=self.settings.name)
        usage = resp.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        return TranslateBatchResult(
            parsed=parsed,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            cached_tokens=int(prompt_details.get("cached_tokens") or 0),
            service_tier=str(resp.get("service_tier") or spec.service_tier or "standard"),
        )

    def generate_json(
        self,
        spec: ModelSpec,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> TranslateBatchResult:
        mode = (self.settings.api_mode or "responses").strip().lower()
        if mode == "chat_completions":
            return self._generate_via_chat_completions(spec, system_prompt=system_prompt, user_prompt=user_prompt)
        return self._generate_via_responses(spec, system_prompt=system_prompt, user_prompt=user_prompt)


def extract_whisper_verbose_words(resp: dict[str, Any]) -> list[TranscribedWord]:
    """Extract chunk-local word (or fallback segment) timestamps from `verbose_json` responses."""
    raw_words = resp.get("words") or []
    words: list[TranscribedWord] = []
    if raw_words:
        for item in raw_words:
            if not isinstance(item, dict):
                continue
            txt = str(item.get("word") or item.get("text") or "").strip()
            if not txt:
                continue
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
            words.append(TranscribedWord(start=start, end=end, text=txt, space_before=True))
        if words:
            return words

    for seg in resp.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        txt = str(seg.get("text") or "").strip()
        if not txt:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        words.append(TranscribedWord(start=start, end=end, text=txt, space_before=True))
    return words


class OpenAICompatTranscribeProvider(TranscribeProvider):
    """OpenAI-compatible `/v1/audio/transcriptions` adapter for models returning native `verbose_json` timestamps."""

    max_chunk_bytes = 24 * 1024 * 1024  # 24 MB headroom below OpenAI/Groq 25 MB upload limit

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        multipart_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(settings)
        self.multipart_fn = multipart_fn or post_multipart

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
        bare = _bare_model_id(model)
        if any(bare == m or bare.startswith(f"{m}-") for m in _UNTIMESTAMPED_STT_MODELS):
            raise ValueError(
                f"模型 `{model}` 不提供原生词级/段级时间戳，暂不适用于高精度字幕转录；"
                "请改用 `elevenlabs/scribe_v3`、`gemini/gemini-3.5-transcribe` 或支持 `verbose_json` 时间戳的 Whisper 兼容模型。"
            )

        opts = options or {}
        api_base = (self.settings.api_base or "https://api.openai.com/v1").rstrip("/")
        url = f"{api_base}/audio/transcriptions"
        fields: list[tuple[str, str]] = [
            ("model", model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ]
        lang_code = str(opts.get("language_code") or "").strip()
        if lang_code:
            fields.append(("language", lang_code))
        prompt = str(opts.get("prompt") or "").strip()
        if prompt:
            fields.append(("prompt", prompt))

        headers = dict(self.settings.extra_headers)
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"

        timeout = max(360, int(self.settings.timeout_seconds or 360))
        last_err: Exception | None = None

        for attempt in range(1, max(1, max_retries) + 1):
            try:
                resp = self.multipart_fn(
                    url,
                    fields=fields,
                    files=[("file", audio_path, "audio/mpeg")],
                    headers=headers,
                    proxy=self.settings.proxy,
                    timeout=timeout,
                    provider_name=self.settings.name,
                )
                words = extract_whisper_verbose_words(resp)
                lang = str(resp.get("language") or "auto")
                dur = resp.get("duration")
                summary = f"语种: {lang}" + (f" | 时长: {float(dur):.1f}s" if dur is not None else "")
                return TranscribeChunkResult(words=words, usage_summary=summary, raw_response=resp)
            except ProviderApiError as api_err:
                last_err = api_err
                if api_err.is_billing_or_quota_exhausted:
                    raise
                if api_err.is_429_rate_limit and attempt < max_retries:
                    wait_s = (api_err.retry_after_seconds + 2) if api_err.retry_after_seconds else min(30, 5 * attempt)
                    if progress_fn:
                        progress_fn(f"    [{self.settings.name} 限流 {attempt}/{max_retries}] 触发 429，等待 {wait_s}s 后重试...")
                    sleep_fn(wait_s)
                    continue
                if api_err.is_503_transient and attempt < max_retries:
                    wait_s = min(30, 8 * attempt)
                    if progress_fn:
                        progress_fn(f"    [{self.settings.name} 高负载 {attempt}/{max_retries}] 遇到 {api_err.code}，{wait_s}s 后重试...")
                    sleep_fn(wait_s)
                    continue
                raise
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    wait_s = min(30, 8 * attempt)
                    if progress_fn:
                        progress_fn(f"    [{self.settings.name} 重试 {attempt}/{max_retries}] 网络/异常 ({type(exc).__name__})，{wait_s}s 后重试...")
                    sleep_fn(wait_s)
                    continue
                raise

        raise RuntimeError(f"{self.settings.name} {model} 调用失败: {last_err}")
