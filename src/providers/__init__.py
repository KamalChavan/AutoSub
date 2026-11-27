"""Provider registry, model-spec parsing, and adapter factory for AutoSub."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from src.config import get_providers_config, resolve_provider_api_key
from src.providers.base import (
    ModelSpec,
    ProviderSettings,
    TranscribeChunkResult,
    TranscribedWord,
    TranscribeProvider,
    TranslateBatchResult,
    TranslateProvider,
)
from src.providers.elevenlabs import ElevenLabsTranscribeProvider
from src.providers.gemini import GeminiTranscribeProvider, GeminiTranslateProvider
from src.providers.openai import OpenAICompatTranscribeProvider, OpenAITranslateProvider
from src.providers.openai_compat import OpenAICompatTranslateProvider

SUPPORTED_PROTOCOLS = ("gemini", "elevenlabs", "openai", "openai_compatible")
TRANSCRIBE_PROTOCOLS = ("gemini", "elevenlabs", "openai", "openai_compatible")
TRANSLATE_PROTOCOLS = ("gemini", "openai", "openai_compatible")


def infer_provider_from_model(model: str, *, default_provider: str = "gemini") -> str:
    """Infer provider name from a bare model identifier when no explicit provider is given."""
    low = (model or "").strip().lower()
    if low.startswith(("gemini-", "google/gemini-", "models/gemini-")):
        return "gemini"
    if low.startswith(("scribe_", "scribe-")):
        return "elevenlabs"
    if low.startswith(("gpt-", "o1", "o3", "o4", "whisper-")):
        return "openai"
    return default_provider


def parse_model_spec(
    raw: str | Mapping[str, Any],
    *,
    default_provider: str = "gemini",
    known_providers: Mapping[str, Any] | None = None,
    provider_override: str | None = None,
    reasoning_effort_override: str | None = None,
) -> ModelSpec:
    """Parse a model string (`openai/gpt-6.2-sol`) or config dict into a normalized `ModelSpec`."""
    known = {str(k).strip().lower() for k in (known_providers or {})} | {
        "gemini",
        "elevenlabs",
        "openai",
        "openai_compatible",
    }

    if isinstance(raw, str):
        raw_provider = ""
        raw_model = raw.strip()
        tier = "standard"
        reasoning_effort = ""
        thinking_level = ""
        temperature: float | None = None
    elif isinstance(raw, Mapping):
        raw_provider = str(raw.get("provider") or "").strip().lower()
        raw_model = str(raw.get("model") or "").strip()
        tier = str(raw.get("service_tier") or "standard").strip().lower() or "standard"
        reasoning_effort = str(raw.get("reasoning_effort") or "").strip().lower()
        thinking_level = str(raw.get("thinking_level") or "").strip()
        raw_temp = raw.get("temperature")
        temperature = float(raw_temp) if raw_temp is not None and str(raw_temp).strip() != "" else None
    else:
        raise ValueError(f"无效的模型配置项: {raw!r}")

    if tier in ("", "default"):
        tier = "standard"

    provider = (provider_override or "").strip().lower() or raw_provider
    model = raw_model

    if not provider and "/" in model:
        prefix, rest = model.split("/", 1)
        pref_low = prefix.strip().lower()
        if pref_low in ("google", "models"):
            provider = "gemini"
            model = rest.strip()
        elif pref_low in known:
            provider = pref_low
            model = rest.strip()
    elif provider == "gemini":
        model = model.removeprefix("google/").removeprefix("models/")

    if not provider:
        provider = infer_provider_from_model(model, default_provider=default_provider)

    if provider == "gemini":
        model = model.removeprefix("google/").removeprefix("models/")

    final_effort = (reasoning_effort_override or "").strip().lower() or reasoning_effort
    return ModelSpec(
        provider=provider,
        model=model,
        service_tier=tier,
        reasoning_effort=final_effort,
        thinking_level=thinking_level,
        temperature=temperature,
    )


def build_provider_settings(
    provider_name: str,
    defaults: dict[str, Any],
    *,
    require_api_key: bool = True,
    project_root: Path | None = None,
    user_config_dir: Path | None = None,
) -> ProviderSettings:
    """Resolve a named provider's protocol, base URL, proxy, and API key from config/env."""
    providers_cfg = get_providers_config(defaults)
    key = (provider_name or "gemini").strip().lower()
    if key not in providers_cfg:
        available = ", ".join(providers_cfg.keys())
        raise ValueError(f"未在配置中找到厂商 `{provider_name}`（可用厂商: {available}）")

    cfg = providers_cfg[key]
    protocol = str(cfg.get("protocol") or key).strip().lower()
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"厂商 `{key}` 使用了不支持的协议 `{protocol}`（支持: {', '.join(SUPPORTED_PROTOCOLS)}）")

    api_base = str(cfg.get("api_base") or "").strip()
    if protocol == "openai_compatible" and not api_base:
        raise ValueError(
            f"厂商 `{key}` (protocol=openai_compatible) 缺少 `api_base`，请在 config/defaults.yaml 的 providers.{key}.api_base 中填写接口地址"
        )

    api_key = resolve_provider_api_key(cfg, project_root=project_root, user_config_dir=user_config_dir)
    if require_api_key and not api_key:
        env_hint = " / ".join(cfg.get("api_key_envs") or ()) or f"{key.upper()}_API_KEY"
        raise RuntimeError(
            f"未检测到 {key} API Key（请设置环境变量 {env_hint}、创建 .env 或填写 config/defaults.yaml 中 providers.{key}.api_key）"
        )

    return ProviderSettings(
        name=key,
        protocol=protocol,
        api_key=api_key,
        api_base=api_base,
        proxy=str(cfg.get("proxy") or "").strip(),
        timeout_seconds=int(cfg.get("timeout_seconds") or 180),
        api_mode=str(cfg.get("api_mode") or "").strip().lower(),
        extra_headers=dict(cfg.get("extra_headers") or {}),
    )


def create_transcribe_provider(
    settings: ProviderSettings,
    *,
    post_fn: Callable[..., dict[str, Any]] | None = None,
    multipart_fn: Callable[..., dict[str, Any]] | None = None,
) -> TranscribeProvider:
    """Instantiate a `TranscribeProvider` for the given resolved settings."""
    proto = settings.protocol
    if proto == "gemini":
        return GeminiTranscribeProvider(settings, post_fn=post_fn)
    if proto == "elevenlabs":
        return ElevenLabsTranscribeProvider(settings, multipart_fn=multipart_fn)
    if proto in ("openai", "openai_compatible"):
        return OpenAICompatTranscribeProvider(settings, multipart_fn=multipart_fn)
    raise ValueError(f"协议 `{proto}` 不支持语音转录（支持: {', '.join(TRANSCRIBE_PROTOCOLS)}）")


def create_translate_provider(
    settings: ProviderSettings,
    *,
    post_fn: Callable[..., dict[str, Any]] | None = None,
) -> TranslateProvider:
    """Instantiate a `TranslateProvider` for the given resolved settings."""
    proto = settings.protocol
    if proto == "gemini":
        return GeminiTranslateProvider(settings, post_fn=post_fn)
    if proto == "openai":
        return OpenAITranslateProvider(settings, post_fn=post_fn)
    if proto == "openai_compatible":
        return OpenAICompatTranslateProvider(settings, post_fn=post_fn)
    raise ValueError(
        f"厂商 `{settings.name}` (protocol={proto}) 仅支持语音转录，不支持字幕翻译（翻译支持协议: {', '.join(TRANSLATE_PROTOCOLS)}）"
    )


def provider_capabilities(protocol: str) -> str:
    """Human-readable capability tag (`STT + LLM`, `STT`, `LLM`) for CLI display."""
    proto = (protocol or "").strip().lower()
    has_stt = proto in TRANSCRIBE_PROTOCOLS
    has_llm = proto in TRANSLATE_PROTOCOLS
    if has_stt and has_llm:
        return "STT + LLM"
    if has_stt:
        return "STT"
    if has_llm:
        return "LLM"
    return "n/a"


__all__ = [
    "ModelSpec",
    "ProviderSettings",
    "SUPPORTED_PROTOCOLS",
    "TRANSCRIBE_PROTOCOLS",
    "TRANSLATE_PROTOCOLS",
    "TranscribeChunkResult",
    "TranscribedWord",
    "TranscribeProvider",
    "TranslateBatchResult",
    "TranslateProvider",
    "build_provider_settings",
    "create_transcribe_provider",
    "create_translate_provider",
    "infer_provider_from_model",
    "parse_model_spec",
    "provider_capabilities",
]
