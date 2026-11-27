"""ElevenLabs Scribe v3 / v2 Speech-to-Text provider adapter (`POST /v1/speech-to-text`)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

from src.http_client import ProviderApiError, post_multipart
from src.providers.base import (
    ProviderSettings,
    TranscribeChunkResult,
    TranscribedWord,
    TranscribeProvider,
)

DEFAULT_ELEVENLABS_TRANSCRIBE_MODEL = "scribe_v3"
_MAX_KEYTERMS = 1000
_MAX_KEYTERM_CHARS = 50
_MAX_KEYTERM_WORDS = 5


def sanitize_keyterms(raw_terms: Sequence[str] | None) -> list[str]:
    """Validate and deduplicate ElevenLabs keyterms (<=50 chars, <=5 words, max 1000)."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_terms or ():
        term = " ".join(str(item or "").replace("\r", " ").replace("\n", " ").split()).strip()
        if not term:
            continue
        if len(term) > _MAX_KEYTERM_CHARS or len(term.split()) > _MAX_KEYTERM_WORDS:
            continue
        if term not in seen:
            seen.add(term)
            cleaned.append(term)
        if len(cleaned) >= _MAX_KEYTERMS:
            break
    return cleaned


def extract_elevenlabs_words(resp: dict[str, Any]) -> list[TranscribedWord]:
    """Parse ElevenLabs Scribe `words[]`, preserving CJK/Latin spacing and optional `speaker_id`."""
    raw_words = resp.get("words") or []
    words: list[TranscribedWord] = []
    pending_space = False
    seen_word = False

    for entry in raw_words:
        if not isinstance(entry, dict):
            continue
        wtype = str(entry.get("type") or "word").strip().lower()
        raw_text = str(entry.get("text") or "")

        if wtype == "audio_event":
            continue
        if wtype == "spacing" or (raw_text and not raw_text.strip()):
            pending_space = True
            continue

        txt = raw_text.strip()
        if not txt:
            continue
        start = float(entry.get("start") or 0.0)
        end = float(entry.get("end") or start)
        speaker_raw = entry.get("speaker_id")
        speaker_id = str(speaker_raw).strip() if speaker_raw is not None and str(speaker_raw).strip() else None

        words.append(
            TranscribedWord(
                start=start,
                end=end,
                text=txt,
                space_before=(pending_space if seen_word else False),
                speaker_id=speaker_id,
            )
        )
        seen_word = True
        pending_space = False

    return words


def build_elevenlabs_form_fields(
    *,
    model: str,
    options: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Build multipart form fields for ElevenLabs `POST /v1/speech-to-text` (repeated array fields)."""
    opts = options or {}
    model_id = (model or DEFAULT_ELEVENLABS_TRANSCRIBE_MODEL).strip()
    is_v3 = model_id.lower().startswith("scribe_v3")

    fields: list[tuple[str, str]] = [
        ("model_id", model_id),
        ("timestamps_granularity", "word"),
        ("tag_audio_events", "true" if opts.get("tag_audio_events") else "false"),
    ]

    lang_code = str(opts.get("language_code") or "").strip()
    if lang_code:
        fields.append(("language_code", lang_code))

    # Scribe v3-only parameters (sending to scribe_v2 returns invalid_parameters)
    if is_v3:
        raw_secondary = opts.get("secondary_language_codes") or []
        if isinstance(raw_secondary, str):
            raw_secondary = [c.strip() for c in raw_secondary.split(",") if c.strip()]
        seen_lang: set[str] = set()
        for code in raw_secondary:
            cleaned = str(code or "").strip()
            if cleaned and cleaned != lang_code and cleaned not in seen_lang:
                seen_lang.add(cleaned)
                fields.append(("secondary_language_codes", cleaned))

        numbers_fmt = str(opts.get("numbers_format") or "").strip().lower()
        if numbers_fmt in ("digits", "words"):
            fields.append(("numbers_format", numbers_fmt))

    if opts.get("no_verbatim"):
        fields.append(("no_verbatim", "true"))

    if opts.get("diarize"):
        fields.append(("diarize", "true"))
        num_speakers = opts.get("num_speakers")
        if num_speakers is not None and int(num_speakers) > 0:
            fields.append(("num_speakers", str(int(num_speakers))))

    # Keyterm prompting adds a surcharge (+$0.05/hr) so only send when explicitly provided
    keyterms = sanitize_keyterms(opts.get("keyterms"))
    for term in keyterms:
        fields.append(("keyterms", term))

    return fields


class ElevenLabsTranscribeProvider(TranscribeProvider):
    """ElevenLabs Scribe v3 / v2 batch transcription adapter with native word-level timestamps."""

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
        api_base = (self.settings.api_base or "https://api.elevenlabs.io/v1").rstrip("/")
        url = f"{api_base}/speech-to-text"
        fields = build_elevenlabs_form_fields(model=model, options=options)
        headers = {"xi-api-key": self.settings.api_key, **self.settings.extra_headers}
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
                    provider_name="ElevenLabs",
                )
                words = extract_elevenlabs_words(resp)
                lang = str(resp.get("language_code") or "auto")
                prob = resp.get("language_probability")
                dur = resp.get("audio_duration_secs")
                meta_parts = [f"语种: {lang}" + (f" ({float(prob):.0%})" if prob is not None else "")]
                if dur is not None:
                    meta_parts.append(f"计费时长: {float(dur):.1f}s")
                return TranscribeChunkResult(
                    words=words,
                    usage_summary=" | ".join(meta_parts),
                    raw_response=resp,
                )
            except ProviderApiError as api_err:
                last_err = api_err
                if api_err.is_billing_or_quota_exhausted:
                    raise ProviderApiError(
                        api_err.code,
                        f"ElevenLabs Credits 不足: {api_err.message}",
                        http_status=api_err.http_status,
                        provider="ElevenLabs",
                        error_type=api_err.error_type,
                    ) from api_err
                if api_err.http_status == 401 and "unusual activity" in api_err.message.lower():
                    raise ProviderApiError(
                        api_err.code,
                        f"{api_err.message}（提示：ElevenLabs Free 档对公共代理/VPN IP 较敏感，可更换独享 IP 或充值 PAYG/订阅）",
                        http_status=401,
                        provider="ElevenLabs",
                        error_type=api_err.error_type,
                    ) from api_err
                if api_err.is_429_rate_limit and attempt < max_retries:
                    wait_s = (api_err.retry_after_seconds + 2) if api_err.retry_after_seconds else min(30, 5 * attempt)
                    if progress_fn:
                        progress_fn(f"    [ElevenLabs 限流/并发 {attempt}/{max_retries}] 触发 429，等待 {wait_s}s 后重试...")
                    sleep_fn(wait_s)
                    continue
                if api_err.is_503_transient and attempt < max_retries:
                    wait_s = min(30, 8 * attempt)
                    if progress_fn:
                        progress_fn(f"    [ElevenLabs 高负载 {attempt}/{max_retries}] 遇到 {api_err.code}，{wait_s}s 后重试...")
                    sleep_fn(wait_s)
                    continue
                raise
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    wait_s = min(30, 8 * attempt)
                    if progress_fn:
                        progress_fn(f"    [ElevenLabs 重试 {attempt}/{max_retries}] 网络/异常 ({type(exc).__name__})，{wait_s}s 后重试...")
                    sleep_fn(wait_s)
                    continue
                raise

        raise RuntimeError(f"ElevenLabs {model} 调用失败: {last_err}")
