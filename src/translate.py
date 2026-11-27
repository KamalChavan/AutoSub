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
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.cache import ResultCache, build_result_cache, make_cache_key, sha256_text
from src.config import DEFAULT_CONFIG_PATH, get_providers_config, load_engine_config
from src.http_client import ProviderApiError, post_json
from src.llm_logger import LlmRequestLogger, build_llm_logger
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
from src.srt import (
    SrtCue,
    align_reference_srt,
    append_srt_cues,
    parse_srt,
    polish_srt_timing,
    wrap_subtitle_line,
    write_srt,
)

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

_SENT_END_RE = re.compile(r"[.!?…。！？][\"'”’）)\]]*$")


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


def split_cues_by_gaps(
    cues: list[SrtCue],
    max_per_batch: int = 280,
    *,
    search_window: int | None = None,
) -> list[list[SrtCue]]:
    """Split cues into the minimum number of batches, cutting at the largest nearby silence pause.

    Searches `±search_window` cues around each ideal even boundary (while strictly enforcing
    `1 <= len(batch) <= max_per_batch`) so batches break at natural speech pauses rather than
    mid-sentence. Falls back to sentence-ending punctuation and even cuts when gaps are uniform.
    """
    if not cues:
        return []
    if max_per_batch < 1:
        raise ValueError("max_per_batch 必须大于 0")
    n = len(cues)
    if n <= max_per_batch:
        return [list(cues)]

    num_batches = math.ceil(n / max_per_batch)
    win = search_window if search_window is not None else min(30, max(5, max_per_batch // 6))

    batches: list[list[SrtCue]] = []
    prev_split = 0
    for m in range(1, num_batches):
        remaining = num_batches - m
        hard_lo = max(prev_split + 1, n - remaining * max_per_batch)
        hard_hi = min(n - remaining, prev_split + max_per_batch)
        ideal = round(m * n / num_batches)
        lo = max(hard_lo, min(hard_hi, ideal - win))
        hi = min(hard_hi, max(hard_lo, ideal + win))

        gaps = {pos: max(0.0, cues[pos].start - cues[pos - 1].end) for pos in range(lo, hi + 1)}
        max_gap = max(gaps.values()) if gaps else 0.0

        best_pos = max(hard_lo, min(hard_hi, ideal))
        best_score: tuple[int, int, int] | None = None
        for pos in range(lo, hi + 1):
            gap = gaps[pos]
            gap_bucket = int(round(gap / 0.05))
            punct = 1 if _SENT_END_RE.search((cues[pos - 1].text or "").strip()) else 0
            dist_pen = -abs(pos - ideal)
            if max_gap >= 0.15:
                score = (gap_bucket, punct, dist_pen)
            else:
                score = (punct, gap_bucket, dist_pen)
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos

        batches.append(cues[prev_split:best_pos])
        prev_split = best_pos

    batches.append(cues[prev_split:])
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


def _parse_id_group(raw_id: Any) -> list[int]:
    """Parse a cue ID (`12`) or merged ID list (`[12, 13]` / `'12-13'`) into unique ordered ints."""
    ids: list[int] = []
    if isinstance(raw_id, (list, tuple)):
        for item in raw_id:
            try:
                val = int(item)
            except (TypeError, ValueError):
                continue
            if val not in ids:
                ids.append(val)
        return ids
    if isinstance(raw_id, str) and "-" in raw_id and not raw_id.strip().startswith("-"):
        parts = [p.strip() for p in raw_id.split("-", 1)]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            a, b = int(parts[0]), int(parts[1])
            if a <= b and (b - a) <= 4:
                return list(range(a, b + 1))
    try:
        return [int(raw_id)]
    except (TypeError, ValueError):
        return []


def _compute_mergeable_pairs(
    batch: list[SrtCue],
    *,
    max_merge_gap: float = 1.8,
    max_merge_duration: float = 6.5,
) -> list[list[int]]:
    """List adjacent cue ID pairs eligible for semantic fragment merging."""
    pairs: list[list[int]] = []
    for i in range(len(batch) - 1):
        a, b = batch[i], batch[i + 1]
        gap = b.start - a.end
        if gap <= max_merge_gap and (b.end - a.start) <= max_merge_duration:
            pairs.append([a.id, b.id])
    return pairs


def _parse_cue_entries_detailed(
    parsed: dict[str, Any],
    batch: list[SrtCue],
    *,
    allow_merge: bool = False,
    max_merge_gap: float = 1.8,
    max_merge_duration: float = 6.5,
    max_merge_cues: int = 3,
) -> tuple[list[dict[str, Any]], int, int, int, list[int]]:
    """Parse `{keep, drop}` supporting optional contiguous cue merges with strict timing & bloat guardrails.

    Returns `(entries, kept_input_cues, dropped_input_cues, merged_groups_count, missing_ids)` ordered by batch position.
    """
    pos_by_id = {cue.id: idx for idx, cue in enumerate(batch)}
    drop_set: set[int] = set()
    for raw_id in parsed.get("drop") or []:
        for cid in _parse_id_group(raw_id):
            if cid in pos_by_id:
                drop_set.add(cid)

    single_map: dict[int, tuple[str, str | None]] = {}
    merge_candidates: list[tuple[list[int], str, str | None]] = []

    def _ingest_keep_row(ids: list[int], target: str, src: str | None) -> None:
        valid_ids = [cid for cid in ids if cid in pos_by_id]
        if not valid_ids or not target:
            return
        if len(valid_ids) == 1:
            single_map[valid_ids[0]] = (target, src)
        elif allow_merge and len(valid_ids) <= max_merge_cues:
            merge_candidates.append((valid_ids, target, src))

    for row in parsed.get("keep") or []:
        if isinstance(row, list) and len(row) >= 2:
            ids = _parse_id_group(row[0])
            target = str(row[1] or "").strip()
            src = str(row[2]).strip() if len(row) >= 3 and row[2] is not None else None
            _ingest_keep_row(ids, target, src)
        elif isinstance(row, dict) and ("id" in row or "ids" in row):
            raw_key = row["ids"] if "ids" in row else row["id"]
            ids = _parse_id_group(raw_key)
            if row.get("drop") is True:
                for cid in ids:
                    if cid in pos_by_id:
                        drop_set.add(cid)
            else:
                target, src = _extract_dict_translation(row)
                _ingest_keep_row(ids, target, src)

    for row in parsed.get("items") or []:
        if not isinstance(row, dict) or ("id" not in row and "ids" not in row):
            continue
        raw_key = row["ids"] if "ids" in row else row["id"]
        ids = _parse_id_group(raw_key)
        if row.get("drop") is True:
            for cid in ids:
                if cid in pos_by_id:
                    drop_set.add(cid)
        else:
            target, src = _extract_dict_translation(row)
            _ingest_keep_row(ids, target, src)

    # Validate merge groups against strict contiguity, silence-gap, and total-duration guardrails
    gap_cap = max_merge_gap + 0.4
    dur_cap = max_merge_duration + 1.0
    merged_by_start_pos: dict[int, dict[str, Any]] = {}
    claimed_positions: set[int] = set()
    merged_groups = 0

    for ids, target, maybe_src in merge_candidates:
        positions = sorted(pos_by_id[cid] for cid in ids)
        if positions != list(range(positions[0], positions[0] + len(positions))):
            continue
        if any(p in claimed_positions for p in positions):
            continue
        group = [batch[p] for p in positions]
        if any((group[k + 1].start - group[k].end) > gap_cap for k in range(len(group) - 1)):
            continue
        if (group[-1].end - group[0].start) > dur_cap:
            continue
        joined_orig = " ".join(c.text for c in group).strip()
        if maybe_src and len(maybe_src) <= int(len(joined_orig) * 1.35) + 15 * len(group):
            src_final = maybe_src
        else:
            src_final = joined_orig
        merged_by_start_pos[positions[0]] = {
            "ids": [c.id for c in group],
            "start": group[0].start,
            "end": group[-1].end,
            "target": target,
            "src": src_final,
        }
        claimed_positions.update(positions)
        merged_groups += 1

    entries: list[dict[str, Any]] = []
    missing_ids: list[int] = []
    kept_input = 0
    dropped_input = 0
    idx = 0
    n = len(batch)
    while idx < n:
        if idx in merged_by_start_pos:
            entry = merged_by_start_pos[idx]
            entries.append(entry)
            span_n = len(entry["ids"])
            kept_input += span_n
            idx += span_n
            continue

        cue = batch[idx]
        if cue.id in single_map:
            target, maybe_src = single_map[cue.id]
            if maybe_src and len(maybe_src) <= int(len(cue.text) * 1.35) + 15:
                src_final = maybe_src
            else:
                src_final = cue.text
            entries.append({
                "ids": [cue.id],
                "start": cue.start,
                "end": cue.end,
                "target": target,
                "src": src_final,
            })
            kept_input += 1
        elif cue.id in drop_set:
            dropped_input += 1
        else:
            missing_ids.append(cue.id)
            entries.append({
                "ids": [cue.id],
                "start": cue.start,
                "end": cue.end,
                "target": cue.text,
                "src": cue.text,
            })
            kept_input += 1
        idx += 1

    return entries, kept_input, dropped_input, merged_groups, missing_ids


def _parse_cue_entries(
    parsed: dict[str, Any],
    batch: list[SrtCue],
    *,
    allow_merge: bool = False,
    max_merge_gap: float = 1.8,
    max_merge_duration: float = 6.5,
    max_merge_cues: int = 3,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Parse `{keep, drop}` supporting optional contiguous cue merges with strict timing & bloat guardrails.

    Returns `(entries, kept_input_cues, dropped_input_cues, merged_groups_count)` ordered by batch position.
    """
    entries, kept, dropped, merged, _missing = _parse_cue_entries_detailed(
        parsed,
        batch,
        allow_merge=allow_merge,
        max_merge_gap=max_merge_gap,
        max_merge_duration=max_merge_duration,
        max_merge_cues=max_merge_cues,
    )
    return entries, kept, dropped, merged


def _parse_compact_translation(
    parsed: dict[str, Any],
    batch: list[SrtCue],
) -> tuple[dict[int, dict[str, Any]], int, int]:
    """Parse `{keep: [[id, translated] | [id, translated, corrected_src]], drop: [id, ...]}`
    with anti-merge `src` length guardrail and missing-ID preservation."""
    entries, kept, dropped, _ = _parse_cue_entries(parsed, batch, allow_merge=False)
    results: dict[int, dict[str, Any]] = {cue.id: {"drop": True} for cue in batch}
    for entry in entries:
        cid = entry["ids"][0]
        results[cid] = {"drop": False, "target": entry["target"], "src": entry["src"]}
    return results, kept, dropped


def format_global_context_section(parsed: dict[str, Any]) -> str:
    """Render Pass-0 extracted video/match context JSON into a compact prompt section."""
    if not isinstance(parsed, dict):
        return ""
    lines: list[str] = []
    summary = str(parsed.get("summary") or "").strip()
    if summary:
        lines.append(f"- Summary: {summary}")
    speakers = [str(s).strip() for s in (parsed.get("speakers_and_names") or []) if str(s).strip()]
    if speakers:
        lines.append(f"- Speakers & Names (do NOT confuse human player/person names with game characters): {', '.join(speakers[:15])}")
    entities = [str(e).strip() for e in (parsed.get("key_entities") or []) if str(e).strip()]
    if entities:
        lines.append(f"- Key Entities / Heroes / Topics: {', '.join(entities[:25])}")
    err_rows: list[str] = []
    for row in (parsed.get("asr_error_map") or [])[:25]:
        if isinstance(row, list) and len(row) >= 2:
            bad, good = str(row[0] or "").strip(), str(row[1] or "").strip()
            if bad and good:
                err_rows.append(f'"{bad}" -> "{good}"')
        elif isinstance(row, dict):
            bad = str(row.get("heard") or row.get("misheard") or row.get("from") or "").strip()
            good = str(row.get("intended") or row.get("fix") or row.get("to") or "").strip()
            if bad and good:
                err_rows.append(f'"{bad}" -> "{good}"')
    if err_rows:
        lines.append(f"- Likely Recurring ASR Typos in This Recording: {'; '.join(err_rows)}")
    if not lines:
        return ""
    return "### Auto-Extracted Full-Video Context (Pass 0)\n" + "\n".join(lines)


def _build_global_context_transcript(cues: list[SrtCue], *, max_chars: int = 24000) -> str:
    """Build a compact transcript for Pass-0 full-video analysis (evenly sampled if very long)."""
    lines = [f"{c.id}: {c.text}" for c in cues if c.text.strip()]
    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full
    # Evenly sample cues across the full timeline to stay within budget
    avg_len = max(12.0, len(full) / max(1, len(lines)))
    target_n = max(40, int(max_chars / avg_len))
    step = len(lines) / target_n
    sampled = [lines[min(len(lines) - 1, int(i * step))] for i in range(target_n)]
    return "\n".join(sampled)[:max_chars]


def build_enhanced_system_prompt(
    base_prompt: str,
    *,
    auto_context_section: str = "",
    allow_merge: bool = False,
    has_alt_asr: bool = False,
) -> str:
    """Append optional Pass-0 context, semantic fragment merging, and dual-ASR disambiguation rules."""
    parts = [base_prompt.rstrip()]
    if auto_context_section.strip():
        parts.append(auto_context_section.strip())
    if allow_merge:
        parts.append(
            "### Semantic Fragment Merging (Optional)\n"
            "- Input JSON may include `mergeable_pairs` (`[[id_a, id_b], ...]`), listing adjacent cues with short gaps.\n"
            "- ONLY when 2-3 consecutive cues connected via `mergeable_pairs` are clearly a single broken sentence, "
            "a split word (e.g. \"Дабл\" + \"ю, дабл ю\"), or a 1-2 word trailing orphan fragment (e.g. \"мне.\" / \"блядь.\"), "
            "you MAY merge them in `\"keep\"` as `[[id1, id2], \"merged translation\"]` or "
            "`[[id1, id2], \"merged translation\", \"corrected merged original\"]`.\n"
            "- NEVER merge cues that are NOT connected in `mergeable_pairs`, nor distinct thoughts or speaker turns. "
            "Every input `id` must still appear in either `\"keep\"` (single or merged) or `\"drop\"` exactly once."
        )
    if has_alt_asr:
        parts.append(
            "### Dual-ASR Reference Disambiguation\n"
            "- Input JSON may include `alt_asr` (`{\"<id>\": \"secondary ASR text\"}`), an independent second ASR transcript "
            "at the same timestamp.\n"
            "- Treat `subtitles` as the authoritative timeline and wording baseline; use `alt_asr` ONLY to disambiguate "
            "phonetic typos, proper nouns, hero/item names, or slang in `subtitles` (and record confirmed fixes in `corrected_src`).\n"
            "- NEVER drop a cue simply because `alt_asr` has no entry for that `id`."
        )
    return "\n\n".join(parts)


def format_cue_lines(
    target_text: str,
    src_text: str,
    *,
    bilingual: bool = True,
    bilingual_order: Sequence[str] = ("target", "source"),
    max_line_chars: float = 0.0,
) -> list[str]:
    rendered_target = (
        wrap_subtitle_line(target_text, max_line_chars=max_line_chars)
        if max_line_chars > 0
        else target_text
    )
    if not bilingual:
        return [rendered_target]
    mapping = {"target": rendered_target, "source": src_text}
    lines = [mapping[key] for key in bilingual_order if key in mapping and mapping[key]]
    return lines or [rendered_target]


def _spec_cache_identity(spec: ModelSpec) -> dict[str, Any]:
    return {
        "provider": spec.provider,
        "model": spec.model,
        "reasoning_effort": spec.reasoning_effort,
        "thinking_level": spec.thinking_level,
        "temperature": spec.temperature,
    }


def _pass0_cache_key(spec: ModelSpec, sys_prompt: str, transcript_text: str) -> str:
    return make_cache_key(
        "pass0",
        {
            "v": "pass0_v1",
            "spec": _spec_cache_identity(spec),
            "system_sha256": sha256_text(sys_prompt),
            "transcript_sha256": sha256_text(transcript_text),
        },
    )


def _translate_batch_cache_key(
    spec: ModelSpec,
    *,
    system_prompt: str,
    user_prompt: str,
    allow_merge: bool,
    max_merge_gap: float,
    max_merge_duration: float,
    repair_missing: bool,
) -> str:
    return make_cache_key(
        "translate",
        {
            "v": "translate_v1",
            "spec": _spec_cache_identity(spec),
            "system_sha256": sha256_text(system_prompt),
            "user_sha256": sha256_text(user_prompt),
            "allow_merge": bool(allow_merge),
            "max_merge_gap": round(float(max_merge_gap), 3),
            "max_merge_duration": round(float(max_merge_duration), 3),
            "repair_missing": bool(repair_missing),
        },
    )


def _build_repair_surrounding_context(
    batch: list[SrtCue],
    entries: list[dict[str, Any]],
    missing_set: set[int],
    *,
    radius: int = 2,
    max_items: int = 24,
) -> list[list[Any]]:
    """Collect already-translated neighbors around missing cue IDs for focused 1-turn repair."""
    pos_by_id = {c.id: idx for idx, c in enumerate(batch)}
    near_positions: set[int] = set()
    for cid in missing_set:
        pos = pos_by_id.get(cid)
        if pos is None:
            continue
        for p in range(max(0, pos - radius), min(len(batch), pos + radius + 1)):
            near_positions.add(p)

    ctx: list[list[Any]] = []
    for entry in entries:
        e_ids = entry.get("ids") or []
        if not e_ids or any(cid in missing_set for cid in e_ids):
            continue
        if any(pos_by_id.get(cid) in near_positions for cid in e_ids):
            label_id: Any = e_ids[0] if len(e_ids) == 1 else e_ids
            ctx.append([label_id, str(entry.get("src") or ""), str(entry.get("target") or "")])
            if len(ctx) >= max_items:
                break
    return ctx


class SubtitleTranslator:
    """Multi-provider fallback translator with Pass-0 context, semantic cue merging, self-repair, cache, 503 retry, and 429 skip."""

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
        cache: ResultCache | None = None,
        llm_logger: LlmRequestLogger | None = None,
    ) -> None:
        self.api_key = api_key
        self.retry_503 = max(1, int(retry_503))
        self.api_base = api_base.rstrip("/")
        self.proxy = proxy
        self.timeout = timeout
        self.sleep_fn = sleep_fn
        self.post_fn = post_fn or post_json
        self.progress_fn = progress_fn
        self.cache = cache
        self.llm_logger = llm_logger

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

    def _call_json_with_fallback(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        task_label: str,
        action_desc: str,
        phase: str = "translate",
        preferred_spec: ModelSpec | None = None,
    ) -> tuple[Any, ModelSpec, float]:
        """Execute a structured JSON LLM call across the fallback chain with 503 retry and 429 skip."""
        prefix = f"[{task_label}] " if task_label else ""
        last_err: Exception | None = None

        ordered_specs: list[ModelSpec] = []
        if preferred_spec is not None:
            ordered_specs.append(preferred_spec)
        for s in self.spec_chain:
            if all(existing.key != s.key for existing in ordered_specs):
                ordered_specs.append(s)

        for spec in ordered_specs:
            if spec.key in self.exhausted_models:
                self._emit(f"  {prefix}跳过当日额度已耗尽模型: {spec.display_label}")
                continue

            adapter = self._get_adapter(spec.provider)
            tpm_retried = False
            attempt = 0
            while attempt < self.retry_503:
                attempt += 1
                self._emit(
                    f"  {prefix}正在调用 {spec.display_label} {action_desc} "
                    f"(第 {attempt}/{self.retry_503} 次尝试)..."
                )
                t0 = time.perf_counter()
                try:
                    res = adapter.generate_json(spec, system_prompt=system_prompt, user_prompt=user_prompt)
                    elapsed = round(time.perf_counter() - t0, 3)
                    if self.llm_logger is not None:
                        self.llm_logger.log_call(
                            phase=phase,
                            task_label=task_label,
                            provider=spec.provider,
                            model=spec.model,
                            service_tier=res.service_tier or spec.service_tier,
                            reasoning_effort=spec.reasoning_effort,
                            attempt=attempt,
                            status="ok",
                            elapsed_ms=int(elapsed * 1000),
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            response=res.parsed,
                            usage={
                                "input_tokens": res.input_tokens,
                                "output_tokens": res.output_tokens,
                                "cached_tokens": res.cached_tokens,
                            },
                        )
                    return res, spec, elapsed
                except ProviderApiError as exc:
                    last_err = exc
                    if self.llm_logger is not None:
                        self.llm_logger.log_call(
                            phase=phase,
                            task_label=task_label,
                            provider=spec.provider,
                            model=spec.model,
                            service_tier=spec.service_tier,
                            reasoning_effort=spec.reasoning_effort,
                            attempt=attempt,
                            status="error",
                            elapsed_ms=int((time.perf_counter() - t0) * 1000),
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            error={"code": exc.code, "http_status": exc.http_status, "message": exc.message},
                        )
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
                    if self.llm_logger is not None:
                        self.llm_logger.log_call(
                            phase=phase,
                            task_label=task_label,
                            provider=spec.provider,
                            model=spec.model,
                            service_tier=spec.service_tier,
                            reasoning_effort=spec.reasoning_effort,
                            attempt=attempt,
                            status="error",
                            elapsed_ms=int((time.perf_counter() - t0) * 1000),
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            error={"code": type(exc).__name__, "http_status": None, "message": str(exc)},
                        )
                    if attempt < self.retry_503:
                        wait_s = 5.0 * attempt
                        self._emit(f"  {prefix}{spec.display_label} 异常 ({type(exc).__name__})，{wait_s:.0f}s 后重试...")
                        self.sleep_fn(wait_s)
                        continue
                    self._emit(f"  {prefix}{spec.display_label} 连续异常，切换下一级模型")
                    break

        raise RuntimeError(f"所有翻译备选模型均失败: {last_err}") from last_err

    def extract_global_context(
        self,
        cues: list[SrtCue],
        *,
        profile_hint: str = "",
        user_vars: Mapping[str, str] | None = None,
    ) -> str:
        """Pass 0: analyze full transcript once to extract speakers, entities, and recurring ASR typos."""
        if not cues:
            return ""
        transcript_text = _build_global_context_transcript(cues)
        var_lines = ", ".join(f"{k}={v}" for k, v in (user_vars or {}).items()) or "(none)"
        sys_prompt = (
            "You are a subtitle pre-analysis assistant. Given a full raw ASR transcript of a video or livestream, "
            "extract compact factual context to help downstream batch translation stay consistent and fix phonetic ASR typos.\n"
            f"Domain hint: {profile_hint or 'General video/livestream'}\n"
            f"Known user metadata (do not contradict): {var_lines}\n"
            "Return ONLY a compact JSON object with keys:\n"
            '  "summary": "1-2 concise sentences summarizing the video/match",\n'
            '  "speakers_and_names": ["Name/alias (role, e.g. streamer or teammate)"],\n'
            '  "key_entities": ["Important heroes, characters, items, projects, or recurring topics"],\n'
            '  "asr_error_map": [["misheard ASR text", "likely intended text (short meaning)"]]\n'
            "Include only high-confidence items actually supported by the transcript. Keep lists concise."
        )
        if self.cache is not None and self.spec_chain:
            primary_key = _pass0_cache_key(self.spec_chain[0], sys_prompt, transcript_text)
            cached = self.cache.get("pass0", primary_key)
            if cached is not None and isinstance(cached.get("section"), str):
                section = cached["section"]
                self.stats.append({
                    "phase": "pass0_context",
                    "provider": str(cached.get("provider") or self.spec_chain[0].provider),
                    "model": str(cached.get("model") or self.spec_chain[0].model),
                    "service_tier": str(cached.get("service_tier") or self.spec_chain[0].service_tier),
                    "cues": len(cues),
                    "kept": 0,
                    "dropped": 0,
                    "elapsed_s": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "cache_hit": True,
                })
                self._emit(
                    f"  [Pass 0 上下文分析] 命中本地缓存 ({cached.get('provider')}/{cached.get('model')}) [cache]"
                )
                return section

        try:
            res, spec, elapsed = self._call_json_with_fallback(
                system_prompt=sys_prompt,
                user_prompt=transcript_text,
                task_label="Pass 0 上下文分析",
                action_desc=f"提取全片上下文（{len(cues)} 条字幕）",
                phase="pass0_context",
            )
            used_tier = res.service_tier or spec.service_tier
            self.stats.append({
                "phase": "pass0_context",
                "provider": spec.provider,
                "model": spec.model,
                "service_tier": used_tier,
                "cues": len(cues),
                "kept": 0,
                "dropped": 0,
                "elapsed_s": elapsed,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "cached_tokens": res.cached_tokens,
            })
            section = format_global_context_section(res.parsed)
            if self.cache is not None and section:
                used_key = _pass0_cache_key(spec, sys_prompt, transcript_text)
                self.cache.set(
                    "pass0",
                    used_key,
                    {
                        "section": section,
                        "provider": spec.provider,
                        "model": spec.model,
                        "service_tier": used_tier,
                    },
                    meta={"provider": spec.provider, "model": spec.model},
                )
            self._emit(
                f"  [Pass 0 上下文分析] 完成 ({spec.provider}/{spec.model}) | 耗时 {elapsed:.1f}s | "
                f"Token: {res.input_tokens} in / {res.output_tokens} out"
            )
            return section
        except Exception as exc:
            self._emit(f"  [Pass 0 上下文分析] 跳过（提取失败，不影响后续翻译）: {exc}")
            return ""

    def translate_batch_entries(
        self,
        batch: list[SrtCue],
        system_prompt: str,
        prev_context: list[list[str]],
        *,
        batch_label: str = "",
        allow_merge: bool = False,
        max_merge_gap: float = 1.8,
        max_merge_duration: float = 6.5,
        ref_asr_map: Mapping[int, str] | None = None,
        repair_missing: bool = True,
    ) -> tuple[list[dict[str, Any]], str, str, int]:
        """Translate one batch and return ordered cue entries (with optional merges, alt_asr, self-repair & cache)."""
        payload_obj: dict[str, Any] = {
            "previous_context": prev_context[-5:],
            "subtitles": [[c.id, c.text] for c in batch],
        }
        if allow_merge:
            pairs = _compute_mergeable_pairs(
                batch,
                max_merge_gap=max_merge_gap,
                max_merge_duration=max_merge_duration,
            )
            if pairs:
                payload_obj["mergeable_pairs"] = pairs
        if ref_asr_map:
            batch_alt = {str(c.id): ref_asr_map[c.id] for c in batch if c.id in ref_asr_map}
            if batch_alt:
                payload_obj["alt_asr"] = batch_alt

        user_prompt = json.dumps(payload_obj, ensure_ascii=False)
        prefix = f"[{batch_label}] " if batch_label else ""

        if self.cache is not None and self.spec_chain:
            primary_spec = self.spec_chain[0]
            primary_key = _translate_batch_cache_key(
                primary_spec,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                allow_merge=allow_merge,
                max_merge_gap=max_merge_gap,
                max_merge_duration=max_merge_duration,
                repair_missing=repair_missing,
            )
            cached = self.cache.get("translate", primary_key)
            if cached is not None and isinstance(cached.get("entries"), list):
                c_entries = cached["entries"]
                c_provider = str(cached.get("provider") or primary_spec.provider)
                c_model = str(cached.get("model") or primary_spec.model)
                c_tier = str(cached.get("service_tier") or primary_spec.service_tier)
                c_kept = int(cached.get("kept", len(c_entries)))
                c_dropped = int(cached.get("dropped", 0))
                c_merged = int(cached.get("merged", 0))
                self.stats.append({
                    "phase": "translate",
                    "provider": c_provider,
                    "model": c_model,
                    "service_tier": c_tier,
                    "cues": len(batch),
                    "kept": c_kept,
                    "dropped": c_dropped,
                    "merged": c_merged,
                    "elapsed_s": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "cache_hit": True,
                })
                merge_part = f" / 合并 {c_merged} 组" if c_merged else ""
                self._emit(
                    f"  {prefix}命中本地缓存 ({c_provider}/{c_model}:{c_tier}) | "
                    f"保留 {c_kept} / 过滤 {c_dropped}{merge_part} [cache]"
                )
                return c_entries, c_model, c_tier, c_merged

        res, spec, elapsed = self._call_json_with_fallback(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            task_label=batch_label,
            action_desc=f"翻译 {len(batch)} 条字幕",
            phase="translate",
        )
        entries, kept, dropped, merged, missing_ids = _parse_cue_entries_detailed(
            res.parsed,
            batch,
            allow_merge=allow_merge,
            max_merge_gap=max_merge_gap,
            max_merge_duration=max_merge_duration,
        )
        used_tier = res.service_tier or spec.service_tier

        still_missing = list(missing_ids)
        if repair_missing and missing_ids:
            missing_set = set(missing_ids)
            missing_cues = [c for c in batch if c.id in missing_set]
            preview_ids = ", ".join(str(i) for i in missing_ids[:8]) + ("..." if len(missing_ids) > 8 else "")
            self._emit(
                f"  {prefix}检测到 {len(missing_ids)} 条漏译/异常合并 ID ([{preview_ids}])，启动 1 轮精确补译..."
            )
            repair_payload: dict[str, Any] = {
                "repair_instruction": (
                    f"Your previous JSON omitted or invalidly merged these subtitle IDs: {missing_ids}. "
                    "Return ONLY a compact JSON object with keys 'keep' and 'drop' accounting for EVERY ID "
                    "in 'missing_subtitles' individually (do NOT merge IDs here)."
                ),
                "surrounding_context": _build_repair_surrounding_context(batch, entries, missing_set),
                "missing_subtitles": [[c.id, c.text] for c in missing_cues],
            }
            if ref_asr_map:
                miss_alt = {str(c.id): ref_asr_map[c.id] for c in missing_cues if c.id in ref_asr_map}
                if miss_alt:
                    repair_payload["alt_asr"] = miss_alt
            try:
                rep_res, rep_spec, rep_elapsed = self._call_json_with_fallback(
                    system_prompt=system_prompt,
                    user_prompt=json.dumps(repair_payload, ensure_ascii=False),
                    task_label=f"{batch_label} 补译" if batch_label else "补译",
                    action_desc=f"精确补译 {len(missing_cues)} 条字幕",
                    phase="repair",
                    preferred_spec=spec,
                )
                rep_entries, _rk, _rd, _rm, rep_still_missing = _parse_cue_entries_detailed(
                    rep_res.parsed,
                    missing_cues,
                    allow_merge=False,
                )
                rep_still_set = set(rep_still_missing)
                rep_by_id = {
                    e["ids"][0]: e
                    for e in rep_entries
                    if len(e.get("ids") or []) == 1 and e["ids"][0] not in rep_still_set
                }
                repaired_kept = 0
                repaired_dropped = 0
                patched_entries: list[dict[str, Any]] = []
                for entry in entries:
                    e_ids = entry.get("ids") or []
                    if len(e_ids) == 1 and e_ids[0] in missing_set:
                        cid = e_ids[0]
                        if cid in rep_by_id:
                            patched_entries.append(rep_by_id[cid])
                            repaired_kept += 1
                        elif cid in rep_still_set:
                            patched_entries.append(entry)
                        else:
                            # Explicitly dropped during repair pass
                            kept -= 1
                            dropped += 1
                            repaired_dropped += 1
                    else:
                        patched_entries.append(entry)
                entries = patched_entries
                still_missing = rep_still_missing
                self.stats.append({
                    "phase": "repair",
                    "provider": rep_spec.provider,
                    "model": rep_spec.model,
                    "service_tier": rep_res.service_tier or rep_spec.service_tier,
                    "cues": len(missing_cues),
                    "kept": repaired_kept,
                    "dropped": repaired_dropped,
                    "merged": 0,
                    "elapsed_s": rep_elapsed,
                    "input_tokens": rep_res.input_tokens,
                    "output_tokens": rep_res.output_tokens,
                    "cached_tokens": rep_res.cached_tokens,
                })
                rep_cache_part = f" ({rep_res.cached_tokens} cached)" if rep_res.cached_tokens else ""
                self._emit(
                    f"  {prefix}补译完成 ({rep_spec.provider}/{rep_spec.model}) | 耗时 {rep_elapsed:.1f}s | "
                    f"补译 {repaired_kept} 条 / 过滤 {repaired_dropped} 条 / 仍缺 {len(still_missing)} 条 | "
                    f"Token: {rep_res.input_tokens} in{rep_cache_part} / {rep_res.output_tokens} out"
                )
            except Exception as rep_exc:
                self._emit(f"  {prefix}补译失败，保留安全原文兜底: {rep_exc}")

        self.stats.append({
            "phase": "translate",
            "provider": spec.provider,
            "model": spec.model,
            "service_tier": used_tier,
            "cues": len(batch),
            "kept": kept,
            "dropped": dropped,
            "merged": merged,
            "elapsed_s": elapsed,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "cached_tokens": res.cached_tokens,
        })

        if self.cache is not None and not still_missing:
            used_key = _translate_batch_cache_key(
                spec,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                allow_merge=allow_merge,
                max_merge_gap=max_merge_gap,
                max_merge_duration=max_merge_duration,
                repair_missing=repair_missing,
            )
            self.cache.set(
                "translate",
                used_key,
                {
                    "provider": spec.provider,
                    "model": spec.model,
                    "service_tier": used_tier,
                    "kept": kept,
                    "dropped": dropped,
                    "merged": merged,
                    "entries": entries,
                },
                meta={"provider": spec.provider, "model": spec.model, "cues": len(batch)},
            )

        cache_part = f" ({res.cached_tokens} cached)" if res.cached_tokens else ""
        merge_part = f" / 合并 {merged} 组" if merged else ""
        self._emit(
            f"  {prefix}成功 ({spec.provider}/{spec.model}:{used_tier}) | 耗时 {elapsed:.1f}s | "
            f"保留 {kept} / 过滤 {dropped}{merge_part} | "
            f"Token: {res.input_tokens} in{cache_part} / {res.output_tokens} out"
        )
        return entries, spec.model, used_tier, merged

    def translate_batch(
        self,
        batch: list[SrtCue],
        system_prompt: str,
        prev_context: list[list[str]],
        *,
        batch_label: str = "",
        repair_missing: bool = True,
    ) -> tuple[dict[int, dict[str, Any]], str, str]:
        """Legacy single-ID batch translation interface (backward compatible)."""
        entries, model, tier, _ = self.translate_batch_entries(
            batch,
            system_prompt,
            prev_context,
            batch_label=batch_label,
            allow_merge=False,
            repair_missing=repair_missing,
        )
        mapped: dict[int, dict[str, Any]] = {cue.id: {"drop": True} for cue in batch}
        for entry in entries:
            cid = entry["ids"][0]
            mapped[cid] = {"drop": False, "target": entry["target"], "src": entry["src"]}
        return mapped, model, tier

    def translate_cues(
        self,
        cues: list[SrtCue],
        *,
        system_prompt: str,
        max_cues_per_batch: int = 280,
        bilingual: bool = True,
        bilingual_order: Sequence[str] = ("target", "source"),
        output_path: Path | None = None,
        allow_merge: bool = False,
        max_merge_gap: float = 1.8,
        max_merge_duration: float = 6.5,
        ref_asr_map: Mapping[int, str] | None = None,
        polish_timing: bool = False,
        auto_context: bool = False,
        profile_hint: str = "",
        user_vars: Mapping[str, str] | None = None,
        repair_missing: bool = True,
        max_line_chars: float = 0.0,
    ) -> list[tuple[float, float, list[str]]]:
        auto_ctx_section = ""
        if auto_context and cues:
            auto_ctx_section = self.extract_global_context(
                cues, profile_hint=profile_hint, user_vars=user_vars,
            )

        effective_prompt = build_enhanced_system_prompt(
            system_prompt,
            auto_context_section=auto_ctx_section,
            allow_merge=allow_merge,
            has_alt_asr=bool(ref_asr_map),
        )

        batches = split_cues_by_gaps(cues, max_per_batch=max_cues_per_batch)
        prev_context: list[list[str]] = []
        output_cues: list[tuple[float, float, list[str]]] = []
        written_count = 0

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")

        for b_idx, batch in enumerate(batches, 1):
            label = f"批次 {b_idx}/{len(batches)}"
            entries, _model, _tier, _merged = self.translate_batch_entries(
                batch,
                effective_prompt,
                prev_context,
                batch_label=label,
                allow_merge=allow_merge,
                max_merge_gap=max_merge_gap,
                max_merge_duration=max_merge_duration,
                ref_asr_map=ref_asr_map,
                repair_missing=repair_missing,
            )
            batch_out: list[tuple[float, float, list[str]]] = []
            for item in entries:
                target = str(item["target"]).strip()
                src = str(item["src"]).strip()
                lines = format_cue_lines(
                    target,
                    src,
                    bilingual=bilingual,
                    bilingual_order=bilingual_order,
                    max_line_chars=max_line_chars,
                )
                entry = (float(item["start"]), float(item["end"]), lines)
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

        if polish_timing and output_cues:
            before_short = sum(1 for s, e, _ in output_cues if (e - s) < 0.7)
            output_cues = polish_srt_timing(output_cues)
            after_short = sum(1 for s, e, _ in output_cues if (e - s) < 0.7)
            if output_path is not None:
                write_srt(output_cues, output_path)
            self._emit(
                f"  [排轴优化] 已完成 CPS/最短驻留时间轴打磨（短于 0.7s 闪现字幕: {before_short} -> {after_short} 条）"
            )

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
    auto_context_override: bool | None = None,
    merge_cues_override: bool | None = None,
    polish_timing_override: bool | None = None,
    repair_missing_override: bool | None = None,
    max_line_chars_override: float | None = None,
    use_cache_override: bool | None = None,
    log_llm_override: bool | None = None,
    ref_srt_path: Path | None = None,
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

    ref_asr_map: dict[int, str] = {}
    if ref_srt_path is not None:
        if not ref_srt_path.is_file():
            raise FileNotFoundError(f"找不到参考 ASR 字幕文件: {ref_srt_path}")
        ref_cues = parse_srt(ref_srt_path)
        ref_asr_map = align_reference_srt(cues, ref_cues)

    allow_merge = (
        bool(merge_cues_override)
        if merge_cues_override is not None
        else bool(profile.translate.get("merge_cues", tl_cfg.get("merge_cues", True)))
    )
    polish_timing = (
        bool(polish_timing_override)
        if polish_timing_override is not None
        else bool(profile.translate.get("polish_timing", tl_cfg.get("polish_timing", True)))
    )
    auto_context = (
        bool(auto_context_override)
        if auto_context_override is not None
        else bool(profile.translate.get("auto_context", tl_cfg.get("auto_context", False)))
    )
    repair_missing = (
        bool(repair_missing_override)
        if repair_missing_override is not None
        else bool(profile.translate.get("repair_missing", tl_cfg.get("repair_missing", True)))
    )
    raw_max_line = (
        max_line_chars_override
        if max_line_chars_override is not None
        else profile.translate.get("max_line_chars", tl_cfg.get("max_line_chars", 24))
    )
    max_line_chars = max(0.0, float(raw_max_line if raw_max_line is not None else 24))
    max_merge_gap = float(tl_cfg.get("max_merge_gap_s", 1.8))
    max_merge_duration = float(tl_cfg.get("max_merge_duration_s", 6.5))

    cache = build_result_cache(defaults, use_cache_override=use_cache_override)
    llm_logger = build_llm_logger(defaults, enabled_override=log_llm_override)

    max_per_batch = int(
        batch_size_override
        or profile.translate.get("max_cues_per_batch")
        or tl_cfg.get("max_cues_per_batch", 280)
    )
    batches = split_cues_by_gaps(cues, max_per_batch=max_per_batch)
    default_suffix = "_bilingual.srt" if bilingual else "_translated.srt"
    out = output_path or input_path.with_name(f"{input_path.stem}{default_suffix}")
    retry_503 = int(tl_cfg.get("retry_503", 3))
    chain_desc = " -> ".join(s.display_label for s in active_specs)

    merged_vars = {**profile.default_vars, **vars_override}
    var_summary = ", ".join(f"{k}={v}" for k, v in merged_vars.items()) or "(无)"
    wrap_desc = f"开(≤{max_line_chars:g}字)" if max_line_chars > 0 else "关"
    feat_flags = [
        f"Pass0上下文={'开' if auto_context else '关'}",
        f"碎句合并={'开' if allow_merge else '关'}",
        f"漏译补译={'开' if repair_missing else '关'}",
        f"排轴优化={'开' if polish_timing else '关'}",
        f"平衡折行={wrap_desc}",
        f"本地缓存={'开' if cache is not None else '关'}",
    ]
    if ref_srt_path is not None:
        feat_flags.append(f"双ASR参考={ref_srt_path.name}({len(ref_asr_map)}处差异)")

    print("=== [AutoSub 字幕翻译] ===", flush=True)
    print(f"配置档位: {profile.name} ({profile.source_path.name})", flush=True)
    print(f"输入字幕: {input_path}（共 {len(cues)} 条，按静音间隙分为 {len(batches)} 批：{[len(b) for b in batches]}）", flush=True)
    print(f"输出路径: {out}（{'双语' if bilingual else '单语'}，每批完成即实时落盘）", flush=True)
    print(f"任务变量: {var_summary} | 术语表: {len(glossary_paths)} 份", flush=True)
    print(f"增强特性: {' | '.join(feat_flags)}", flush=True)
    if llm_logger is not None:
        print(f"调试日志: {llm_logger.log_path}", flush=True)
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
        cache=cache,
        llm_logger=llm_logger,
    )
    translated = translator.translate_cues(
        cues,
        system_prompt=system_prompt,
        max_cues_per_batch=max_per_batch,
        bilingual=bilingual,
        bilingual_order=profile.bilingual_order,
        output_path=out,
        allow_merge=allow_merge,
        max_merge_gap=max_merge_gap,
        max_merge_duration=max_merge_duration,
        ref_asr_map=ref_asr_map or None,
        polish_timing=polish_timing,
        auto_context=auto_context,
        profile_hint=profile.description or profile.name,
        user_vars=merged_vars,
        repair_missing=repair_missing,
        max_line_chars=max_line_chars,
    )
    total_in = sum(int(s.get("input_tokens", 0)) for s in translator.stats)
    total_out = sum(int(s.get("output_tokens", 0)) for s in translator.stats)
    total_cached = sum(int(s.get("cached_tokens", 0)) for s in translator.stats)
    total_dropped = sum(int(s.get("dropped", 0)) for s in translator.stats if s.get("phase") == "translate")
    total_merged = sum(int(s.get("merged", 0)) for s in translator.stats if s.get("phase") == "translate")
    cache_hits = sum(1 for s in translator.stats if s.get("phase") == "translate" and s.get("cache_hit"))
    total_s = sum(float(s.get("elapsed_s", 0.0)) for s in translator.stats)
    cache_part = f" ({total_cached} cached)" if total_cached else ""
    merge_summary = f" / 合并 {total_merged} 组" if total_merged else ""
    hit_summary = f" | 本地缓存命中 {cache_hits} 批" if cache_hits else ""
    print(
        f"[翻译完成] {out} | 输入 {len(cues)} 条 -> 输出 {len(translated)} 条 (过滤 {total_dropped} 条{merge_summary}) | "
        f"纯推理耗时 {total_s:.1f}s | Token: {total_in} in{cache_part} / {total_out} out{hit_summary}",
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
    auto_context_override: bool | None = None,
    merge_cues_override: bool | None = None,
    polish_timing_override: bool | None = None,
    repair_missing_override: bool | None = None,
    max_line_chars_override: float | None = None,
    use_cache_override: bool | None = None,
    log_llm_override: bool | None = None,
    ref_srt_path: Path | None = None,
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
            auto_context_override=auto_context_override,
            merge_cues_override=merge_cues_override,
            polish_timing_override=polish_timing_override,
            repair_missing_override=repair_missing_override,
            max_line_chars_override=max_line_chars_override,
            use_cache_override=use_cache_override,
            log_llm_override=log_llm_override,
            ref_srt_path=ref_srt_path,
            output_path=output_path,
        )
        return 0
    except Exception as exc:
        print(f"[错误] {exc}", flush=True)
        return 1
