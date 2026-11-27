"""Profile-driven multi-provider subtitle translation & denoising pipeline (`autosub translate`).

Domain-agnostic engine: prompts, terminology glossaries, target language, bilingual/monolingual
layout, and context variables are all loaded from `profiles/*.yaml` (or any external YAML file)
and overridden via `--var key=value`.

Features:
- Even batch splitting (avoiding tiny tail batches)
- Cross-provider fallback chain (e.g. `gemini`, `openai/gpt-6.2-sol`, `openai_compatible`)
- 3 in-place attempts on 503 before stepping down
- Per-run RPD/quota exhaustion skipping on 429
- Compact `{keep, drop}` JSON parsing with missing-ID safety & `src` bloat guardrail
- Real-time stdout progress + incremental per-batch `.srt` disk flushing
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.config import DEFAULT_CONFIG_PATH, get_providers_config, load_engine_config
from src.http_client import ProviderApiError, post_json
from src.profile import (
    load_profile,
    merge_glossaries,
    parse_var_overrides,
    render_glossary_section,
    render_prompt,
)
from src.providers import (
    ModelSpec,
    ProviderSettings,
    TranslateProvider,
    build_provider_settings,
    create_translate_provider,
    parse_model_spec,
)
from src.srt import SrtCue, append_srt_cues, parse_srt

DEFAULT_TRANSLATE_MODELS: tuple[dict[str, str], ...] = (
    {"provider": "gemini", "model": "gemini-4-hydrogen", "service_tier": "standard"},
    {"provider": "gemini", "model": "gemini-3.8-flash", "service_tier": "standard"},
    {"provider": "gemini", "model": "gemini-3.5-flash-lite", "service_tier": "standard"},
)

_DEFAULT_MODELS_BY_PROTOCOL: dict[str, list[dict[str, str]]] = {
    "gemini": [dict(m) for m in DEFAULT_TRANSLATE_MODELS],
    "openai": [
        {"provider": "openai", "model": "gpt-6.2-sol", "reasoning_effort": "low"},
        {"provider": "openai", "model": "gpt-6-luna", "reasoning_effort": "low"},
    ],
}


def split_cues_evenly(cues: list[SrtCue], max_per_batch: int = 280) -> list[list[SrtCue]]:
    """Split cues evenly into the minimum number of batches (e.g. 543 / 280 -> [272, 271])."""
    if not cues:
        return []
    if max_per_batch < 1:
        raise ValueError("max_per_batch 必须大于 0")
    num_batches = math.ceil(len(cues) / max_per_batch)
    base, extra = divmod(len(cues), num_batches)
    batches: list[list[SrtCue]] = []
    cursor = 0
    for i in range(num_batches):
        size = base + (1 if i < extra else 0)
        batches.append(cues[cursor : cursor + size])
        cursor += size
    return batches


def normalize_translate_model_specs(
    models: Sequence[Any] | None = DEFAULT_TRANSLATE_MODELS,
    *,
    default_provider: str = "gemini",
    known_providers: Mapping[str, Any] | None = None,
    provider_override: str | None = None,
    reasoning_effort_override: str | None = None,
) -> list[ModelSpec]:
    """Normalize raw model configs or strings into deduplicated `ModelSpec` entries."""
    specs: list[ModelSpec] = []
    seen: set[tuple[str, str, str]] = set()
    for item in models or DEFAULT_TRANSLATE_MODELS:
        if not isinstance(item, (str, Mapping)):
            continue
        spec = parse_model_spec(
            item,
            default_provider=default_provider,
            known_providers=known_providers,
            provider_override=provider_override,
            reasoning_effort_override=reasoning_effort_override,
        )
        if not spec.model:
            continue
        if spec.key not in seen:
            seen.add(spec.key)
            specs.append(spec)
    return specs


def normalize_translate_chain(
    models: list[Any] | tuple[Any, ...] | None = DEFAULT_TRANSLATE_MODELS,
) -> list[tuple[str, str]]:
    """Legacy `(model, service_tier)` view of the translation chain."""
    chain: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for spec in normalize_translate_model_specs(models):
        key = (spec.model, spec.service_tier)
        if key not in seen:
            seen.add(key)
            chain.append(key)
    return chain


def _extract_dict_translation(row: dict[str, Any]) -> tuple[str, str | None]:
    for key in ("target", "translation", "text", "zh"):
        val = row.get(key)
        if val is not None and str(val).strip():
            src_val = row.get("src") or row.get("source")
            return str(val).strip(), str(src_val).strip() if src_val else None
    return "", None


def _parse_compact_translation(
    parsed: dict[str, Any],
    batch: list[SrtCue],
) -> tuple[dict[int, dict[str, Any]], int, int]:
    """Parse `{keep: [[id, translated] | [id, translated, corrected_src]], drop: [id, ...]}`
    with anti-merge `src` length guardrail and missing-ID preservation."""
    drop_set: set[int] = set()
    for raw_id in parsed.get("drop") or []:
        try:
            drop_set.add(int(raw_id))
        except (TypeError, ValueError):
            continue

    keep_map: dict[int, tuple[str, str | None]] = {}
    for row in parsed.get("keep") or []:
        if isinstance(row, list) and len(row) >= 2:
            try:
                cid = int(row[0])
            except (TypeError, ValueError):
                continue
            target = str(row[1] or "").strip()
            src = str(row[2]).strip() if len(row) >= 3 and row[2] is not None else None
            if target:
                keep_map[cid] = (target, src)
        elif isinstance(row, dict) and "id" in row:
            try:
                cid = int(row["id"])
            except (TypeError, ValueError):
                continue
            if row.get("drop") is True:
                drop_set.add(cid)
            else:
                target, src = _extract_dict_translation(row)
                if target:
                    keep_map[cid] = (target, src)

    for row in parsed.get("items") or []:
        if not isinstance(row, dict) or "id" not in row:
            continue
        try:
            cid = int(row["id"])
        except (TypeError, ValueError):
            continue
        if row.get("drop") is True:
            drop_set.add(cid)
        else:
            target, src = _extract_dict_translation(row)
            if target:
                keep_map[cid] = (target, src)

    results: dict[int, dict[str, Any]] = {}
    kept = dropped = 0
    for cue in batch:
        if cue.id in keep_map:
            target, maybe_src = keep_map[cue.id]
            if maybe_src and len(maybe_src) <= int(len(cue.text) * 1.35) + 15:
                src_final = maybe_src
            else:
                src_final = cue.text
            results[cue.id] = {"drop": False, "target": target, "src": src_final}
            kept += 1
        elif cue.id in drop_set:
            results[cue.id] = {"drop": True}
            dropped += 1
        else:
            results[cue.id] = {"drop": False, "target": cue.text, "src": cue.text}
            kept += 1
    return results, kept, dropped


def format_cue_lines(
    target_text: str,
    src_text: str,
    *,
    bilingual: bool = True,
    bilingual_order: Sequence[str] = ("target", "source"),
) -> list[str]:
    if not bilingual:
        return [target_text]
    mapping = {"target": target_text, "source": src_text}
    lines = [mapping[key] for key in bilingual_order if key in mapping and mapping[key]]
    return lines or [target_text]


class SubtitleTranslator:
    """Multi-provider fallback translator with 503 3-attempt retry, 429 quota skip, and incremental SRT flush."""

    def __init__(
        self,
        *,
        api_key: str = "",
        models: Sequence[Any] | None = DEFAULT_TRANSLATE_MODELS,
        retry_503: int = 3,
        api_base: str = "https://generativelanguage.googleapis.com/v1beta",
        proxy: str = "",
        timeout: int = 180,
        providers_map: Mapping[str, ProviderSettings] | None = None,
        default_provider: str = "gemini",
        reasoning_effort_override: str | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        post_fn: Callable[..., dict[str, Any]] | None = None,
        progress_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.api_key = api_key
        self.retry_503 = max(1, int(retry_503))
        self.api_base = api_base.rstrip("/")
        self.proxy = proxy
        self.timeout = timeout
        self.sleep_fn = sleep_fn
        self.post_fn = post_fn or post_json
        self.progress_fn = progress_fn

        self.providers_map: dict[str, ProviderSettings] = dict(providers_map or {})
        if not self.providers_map:
            self.providers_map["gemini"] = ProviderSettings(
                name="gemini",
                protocol="gemini",
                api_key=api_key,
                api_base=self.api_base,
                proxy=proxy,
                timeout_seconds=timeout,
            )

        self.spec_chain: list[ModelSpec] = normalize_translate_model_specs(
            models,
            default_provider=default_provider,
            known_providers=self.providers_map,
            reasoning_effort_override=reasoning_effort_override,
        )
        self.chain: list[tuple[str, str]] = [(s.model, s.service_tier) for s in self.spec_chain]
        self._adapters: dict[str, TranslateProvider] = {}
        self.exhausted_models: set[tuple[str, str, str]] = set()
        self.stats: list[dict[str, Any]] = []

    def _emit(self, message: str) -> None:
        if self.progress_fn is not None:
            self.progress_fn(message)

    def _get_adapter(self, provider_name: str) -> TranslateProvider:
        if provider_name not in self._adapters:
            settings = self.providers_map.get(provider_name)
            if settings is None:
                raise RuntimeError(f"未配置翻译厂商 `{provider_name}` 的连接参数")
            self._adapters[provider_name] = create_translate_provider(settings, post_fn=self.post_fn)
        return self._adapters[provider_name]

    def translate_batch(
        self,
        batch: list[SrtCue],
        system_prompt: str,
        prev_context: list[list[str]],
        *,
        batch_label: str = "",
    ) -> tuple[dict[int, dict[str, Any]], str, str]:
        user_prompt = json.dumps(
            {
                "previous_context": prev_context[-5:],
                "subtitles": [[c.id, c.text] for c in batch],
            },
            ensure_ascii=False,
        )
        prefix = f"[{batch_label}] " if batch_label else ""
        last_err: Exception | None = None

        for spec in self.spec_chain:
            if spec.key in self.exhausted_models:
                self._emit(f"  {prefix}跳过当日额度已耗尽模型: {spec.display_label}")
                continue

            adapter = self._get_adapter(spec.provider)
            tpm_retried = False
            attempt = 0
            while attempt < self.retry_503:
                attempt += 1
                self._emit(
                    f"  {prefix}正在调用 {spec.display_label} 翻译 {len(batch)} 条字幕 "
                    f"(第 {attempt}/{self.retry_503} 次尝试)..."
                )
                try:
                    t0 = time.perf_counter()
                    res = adapter.generate_json(spec, system_prompt=system_prompt, user_prompt=user_prompt)
                    elapsed = round(time.perf_counter() - t0, 3)
                    mapped, kept, dropped = _parse_compact_translation(res.parsed, batch)
                    used_tier = res.service_tier or spec.service_tier
                    self.stats.append({
                        "provider": spec.provider,
                        "model": spec.model,
                        "service_tier": used_tier,
                        "cues": len(batch),
                        "kept": kept,
                        "dropped": dropped,
                        "elapsed_s": elapsed,
                        "input_tokens": res.input_tokens,
                        "output_tokens": res.output_tokens,
                        "cached_tokens": res.cached_tokens,
                    })
                    cache_part = f" ({res.cached_tokens} cached)" if res.cached_tokens else ""
                    self._emit(
                        f"  {prefix}成功 ({spec.provider}/{spec.model}:{used_tier}) | 耗时 {elapsed:.1f}s | "
                        f"保留 {kept} / 过滤 {dropped} | "
                        f"Token: {res.input_tokens} in{cache_part} / {res.output_tokens} out"
                    )
                    return mapped, spec.model, used_tier
                except ProviderApiError as exc:
                    last_err = exc
                    if exc.is_429_rate_limit or exc.is_billing_or_quota_exhausted:
                        if (
                            not exc.is_daily_quota_exhausted
                            and not tpm_retried
                            and exc.retry_after_seconds is not None
                        ):
                            tpm_retried = True
                            attempt -= 1
                            wait_s = exc.retry_after_seconds + 2
                            self._emit(f"  {prefix}{spec.display_label} 触发短时限流 (429)，等待 {wait_s}s 后重试...")
                            self.sleep_fn(wait_s)
                            continue
                        self.exhausted_models.add(spec.key)
                        self._emit(f"  {prefix}{spec.display_label} 额度耗尽 ({exc.code})，本轮跳过并切换下一级模型")
                        break
                    if exc.is_503_transient:
                        if attempt < self.retry_503:
                            wait_s = 5.0 * attempt
                            self._emit(
                                f"  {prefix}{spec.display_label} 遇到 {exc.code} 高负载，"
                                f"{wait_s:.0f}s 后原地重试 ({attempt}/{self.retry_503})..."
                            )
                            self.sleep_fn(wait_s)
                            continue
                        self._emit(
                            f"  {prefix}{spec.display_label} 连续 {self.retry_503} 次 {exc.code} 失败，切换下一级模型"
                        )
                        break
                    self._emit(f"  {prefix}{spec.display_label} 报错 ({exc.code})，切换下一级模型")
                    break
                except Exception as exc:
                    last_err = exc
                    if attempt < self.retry_503:
                        wait_s = 5.0 * attempt
                        self._emit(f"  {prefix}{spec.display_label} 异常 ({type(exc).__name__})，{wait_s:.0f}s 后重试...")
                        self.sleep_fn(wait_s)
                        continue
                    self._emit(f"  {prefix}{spec.display_label} 连续异常，切换下一级模型")
                    break

        raise RuntimeError(f"所有翻译备选模型均失败: {last_err}") from last_err

    def translate_cues(
        self,
        cues: list[SrtCue],
        *,
        system_prompt: str,
        max_cues_per_batch: int = 280,
        bilingual: bool = True,
        bilingual_order: Sequence[str] = ("target", "source"),
        output_path: Path | None = None,
    ) -> list[tuple[float, float, list[str]]]:
        batches = split_cues_evenly(cues, max_per_batch=max_cues_per_batch)
        prev_context: list[list[str]] = []
        output_cues: list[tuple[float, float, list[str]]] = []
        written_count = 0

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")

        for b_idx, batch in enumerate(batches, 1):
            label = f"批次 {b_idx}/{len(batches)}"
            mapped, _model, _tier = self.translate_batch(
                batch, system_prompt, prev_context, batch_label=label,
            )
            batch_out: list[tuple[float, float, list[str]]] = []
            for cue in batch:
                item = mapped[cue.id]
                if item.get("drop"):
                    continue
                target = str(item["target"]).strip()
                src = str(item["src"]).strip()
                lines = format_cue_lines(
                    target, src, bilingual=bilingual, bilingual_order=bilingual_order,
                )
                entry = (cue.start, cue.end, lines)
                batch_out.append(entry)
                output_cues.append(entry)
                prev_context.append([src, target])

            if output_path is not None and batch_out:
                written_count = append_srt_cues(
                    batch_out,
                    output_path,
                    start_index=written_count + 1,
                    truncate=(written_count == 0),
                )
                self._emit(f"  [{label}] 已实时写入 {output_path.name}（累计 {written_count} 条）")

        return output_cues


def _resolve_translate_models_and_providers(
    defaults: dict[str, Any],
    tl_cfg: dict[str, Any],
    profile_tl_cfg: dict[str, Any],
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
) -> tuple[list[ModelSpec], dict[str, ProviderSettings]]:
    providers_cfg = get_providers_config(defaults)
    default_provider = str(tl_cfg.get("provider") or "gemini").strip().lower()

    if model_override:
        raw_models: list[Any] = [m.strip() for m in model_override.split(",") if m.strip()]
        specs = normalize_translate_model_specs(
            raw_models,
            default_provider=(provider_override or default_provider).strip().lower(),
            known_providers=providers_cfg,
            provider_override=provider_override,
            reasoning_effort_override=reasoning_effort_override,
        )
    elif provider_override:
        ov = provider_override.strip().lower()
        proto = str((providers_cfg.get(ov) or {}).get("protocol") or ov).strip().lower()
        preset = _DEFAULT_MODELS_BY_PROTOCOL.get(proto)
        if not preset:
            raise ValueError(
                f"厂商 `{ov}` 没有内置默认翻译模型，请同时通过 `--model` / `--translate-model` 指定模型名"
            )
        specs = normalize_translate_model_specs(
            preset,
            default_provider=ov,
            known_providers=providers_cfg,
            provider_override=ov,
            reasoning_effort_override=reasoning_effort_override,
        )
    else:
        raw_cfg_models = profile_tl_cfg.get("models") or tl_cfg.get("models") or list(DEFAULT_TRANSLATE_MODELS)
        specs = normalize_translate_model_specs(
            raw_cfg_models,
            default_provider=default_provider,
            known_providers=providers_cfg,
            reasoning_effort_override=reasoning_effort_override,
        )

    if not specs:
        raise ValueError("翻译模型链为空，请检查 config/defaults.yaml 或 --model 参数")

    unique_providers = list(dict.fromkeys(s.provider for s in specs))
    providers_map: dict[str, ProviderSettings] = {}
    missing_errors: list[str] = []

    for pname in unique_providers:
        try:
            settings = build_provider_settings(pname, defaults, require_api_key=True)
            # Validate protocol supports translation early
            create_translate_provider(settings)
            providers_map[pname] = settings
        except ValueError:
            raise
        except RuntimeError as exc:
            missing_errors.append(str(exc))

    if not providers_map:
        raise RuntimeError("；".join(missing_errors) or "未检测到任何可用翻译厂商的 API Key")

    active_specs = [s for s in specs if s.provider in providers_map]
    skipped = [p for p in unique_providers if p not in providers_map]
    if skipped:
        print(f"[提示] 以下翻译厂商未配置 API Key，本轮自动跳过: {', '.join(skipped)}", flush=True)
    return active_specs, providers_map


def translate_srt(
    input_path: Path,
    *,
    defaults_path: Path = DEFAULT_CONFIG_PATH,
    profile_name: str | None = None,
    vars_list: list[str] | None = None,
    extra_glossaries: list[str] | None = None,
    target_language_override: str | None = None,
    bilingual_override: bool | None = None,
    batch_size_override: int | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
    output_path: Path | None = None,
) -> Path:
    """Translate an original-language `.srt` file using a profile and optional `--var` overrides."""
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入字幕文件: {input_path}")
    if input_path.suffix.lower() != ".srt":
        raise ValueError(
            f"`translate` 仅接受 .srt 文件；视频/音频请先运行 `python3 main.py transcript {input_path.name}` 或 `run`"
        )

    defaults = load_engine_config(defaults_path)
    tl_cfg = defaults.get("translate") or {}

    effective_profile_name = profile_name or str(tl_cfg.get("default_profile") or "default")
    profile = load_profile(effective_profile_name)

    active_specs, providers_map = _resolve_translate_models_and_providers(
        defaults,
        tl_cfg,
        profile.translate,
        provider_override=provider_override,
        model_override=model_override,
        reasoning_effort_override=reasoning_effort_override,
    )

    vars_override = parse_var_overrides(vars_list)
    if target_language_override:
        vars_override["target_language"] = target_language_override

    glossary_paths = list(profile.glossaries)
    for extra in extra_glossaries or []:
        glossary_paths.append(Path(extra).resolve())
    glossary_text = render_glossary_section(merge_glossaries(glossary_paths)) if glossary_paths else ""

    bilingual = profile.bilingual if bilingual_override is None else bilingual_override
    cues = parse_srt(input_path)
    if not cues:
        raise ValueError(f"字幕文件为空或格式无效: {input_path}")

    max_per_batch = int(
        batch_size_override
        or profile.translate.get("max_cues_per_batch")
        or tl_cfg.get("max_cues_per_batch", 280)
    )
    batches = split_cues_evenly(cues, max_per_batch=max_per_batch)
    default_suffix = "_bilingual.srt" if bilingual else "_translated.srt"
    out = output_path or input_path.with_name(f"{input_path.stem}{default_suffix}")
    retry_503 = int(tl_cfg.get("retry_503", 3))
    chain_desc = " -> ".join(s.display_label for s in active_specs)

    merged_vars = {**profile.default_vars, **vars_override}
    var_summary = ", ".join(f"{k}={v}" for k, v in merged_vars.items()) or "(无)"

    print("=== [AutoSub 字幕翻译] ===", flush=True)
    print(f"配置档位: {profile.name} ({profile.source_path.name})", flush=True)
    print(f"输入字幕: {input_path}（共 {len(cues)} 条，均匀分为 {len(batches)} 批：{[len(b) for b in batches]}）", flush=True)
    print(f"输出路径: {out}（{'双语' if bilingual else '单语'}，每批完成即实时落盘）", flush=True)
    print(f"任务变量: {var_summary} | 术语表: {len(glossary_paths)} 份", flush=True)
    print(f"模型链: {chain_desc}（503 原地重试 {retry_503} 次后切换）", flush=True)

    system_prompt = render_prompt(profile, vars_override=vars_override, glossary_text=glossary_text)
    models_dicts = [
        {
            "provider": s.provider,
            "model": s.model,
            "service_tier": s.service_tier,
            "reasoning_effort": s.reasoning_effort,
            "thinking_level": s.thinking_level,
            "temperature": s.temperature,
        }
        for s in active_specs
    ]
    translator = SubtitleTranslator(
        models=models_dicts,
        retry_503=retry_503,
        providers_map=providers_map,
        progress_fn=lambda msg: print(msg, flush=True),
    )
    translated = translator.translate_cues(
        cues,
        system_prompt=system_prompt,
        max_cues_per_batch=max_per_batch,
        bilingual=bilingual,
        bilingual_order=profile.bilingual_order,
        output_path=out,
    )
    total_in = sum(int(s.get("input_tokens", 0)) for s in translator.stats)
    total_out = sum(int(s.get("output_tokens", 0)) for s in translator.stats)
    total_cached = sum(int(s.get("cached_tokens", 0)) for s in translator.stats)
    total_s = sum(float(s.get("elapsed_s", 0.0)) for s in translator.stats)
    cache_part = f" ({total_cached} cached)" if total_cached else ""
    print(
        f"[翻译完成] {out} | 输入 {len(cues)} 条 -> 保留 {len(translated)} 条 (过滤 {len(cues) - len(translated)} 条) | "
        f"纯推理耗时 {total_s:.1f}s | Token: {total_in} in{cache_part} / {total_out} out",
        flush=True,
    )
    return out


def run_translate_cli(
    input_path: Path,
    *,
    defaults_path: Path = DEFAULT_CONFIG_PATH,
    profile_name: str | None = None,
    vars_list: list[str] | None = None,
    extra_glossaries: list[str] | None = None,
    target_language_override: str | None = None,
    bilingual_override: bool | None = None,
    batch_size_override: int | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
    output_path: Path | None = None,
) -> int:
    try:
        translate_srt(
            input_path,
            defaults_path=defaults_path,
            profile_name=profile_name,
            vars_list=vars_list,
            extra_glossaries=extra_glossaries,
            target_language_override=target_language_override,
            bilingual_override=bilingual_override,
            batch_size_override=batch_size_override,
            provider_override=provider_override,
            model_override=model_override,
            reasoning_effort_override=reasoning_effort_override,
            output_path=output_path,
        )
        return 0
    except Exception as exc:
        print(f"[错误] {exc}", flush=True)
        return 1
