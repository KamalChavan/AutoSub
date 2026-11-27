"""Word-anchor LLM semantic re-segmentation (`autosub resegment` & integrated `translate`/`run` pre-pass).

Takes original-timeline word timestamps from `.words.json`, partitions speech at hard acoustic
boundaries (long silence pauses & speaker turns), sends only multi-clause/long/paused candidate
blocks to the configured LLM to return compact ending word IDs (`{"splits": {"1": [19, 26]}}`),
validates length/duration invariants with 1-turn targeted self-repair, and reconstructs exact
drift-free `SrtCue` timelines.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from src.cache import ResultCache, build_result_cache, make_cache_key, sha256_text
from src.config import DEFAULT_CONFIG_PATH, load_engine_config
from src.llm_logger import build_llm_logger
from src.profile import load_profile
from src.providers import ModelSpec
from src.srt import (
    SrtCue,
    _effective_display_chars,
    _join_word_buf,
    clamp_stretched_words,
    find_sidecar_words_json,
    load_words_json,
    reconstruct_cues_from_word_spans,
    write_srt,
)

_SENT_END_WORD_RE = re.compile(r"[.!?…。！？][\"'”’）)\]]*$")
_CLAUSE_PUNCT_WORD_RE = re.compile(r"[,;:—–，；：、][\"'”’）)\]]*$")


@dataclass
class WordBlock:
    """Acoustically contiguous speech block of sequential words (`first_wid..last_wid`)."""

    block_id: int
    words: list[dict[str, Any]]
    first_wid: int
    last_wid: int
    needs_llm: bool


def _split_oversized_group(
    group: list[dict[str, Any]],
    *,
    max_block_words: int = 80,
) -> list[list[dict[str, Any]]]:
    """Recursively divide an abnormally long continuous speech group at its largest internal pause."""
    n = len(group)
    if n <= max_block_words:
        return [group]
    mid = n // 2
    lo = max(1, mid - max(8, n // 4))
    hi = min(n - 1, mid + max(8, n // 4))
    best_pos = mid
    best_score: tuple[int, int, int] | None = None
    for pos in range(lo, hi + 1):
        gap = max(0.0, float(group[pos]["start"]) - float(group[pos - 1]["end"]))
        gap_bucket = int(round(gap / 0.05))
        punct = 2 if _SENT_END_WORD_RE.search(str(group[pos - 1].get("text") or "")) else (
            1 if _CLAUSE_PUNCT_WORD_RE.search(str(group[pos - 1].get("text") or "")) else 0
        )
        score = (gap_bucket, punct, -abs(pos - mid))
        if best_score is None or score > best_score:
            best_score = score
            best_pos = pos
    return (
        _split_oversized_group(group[:best_pos], max_block_words=max_block_words)
        + _split_oversized_group(group[best_pos:], max_block_words=max_block_words)
    )


def _block_needs_llm(
    buf: list[dict[str, Any]],
    *,
    max_cue_chars: float = 28.0,
    max_cue_duration_s: float = 5.5,
) -> bool:
    """Return True if an acoustic block could benefit from semantic splitting or internal pause judgment."""
    if len(buf) <= 1:
        return False
    text = _join_word_buf(buf)
    eff_chars = _effective_display_chars(text)
    dur = float(buf[-1]["end"]) - float(buf[0]["start"])
    if eff_chars > max_cue_chars or dur > min(max_cue_duration_s, 4.2) or len(buf) >= 10:
        return True
    sent_positions = [
        idx for idx, w in enumerate(buf)
        if _SENT_END_WORD_RE.search(str(w.get("text") or "").strip())
    ]
    if len(sent_positions) >= 2:
        return True
    if len(sent_positions) == 1 and sent_positions[0] < len(buf) - 1:
        return True
    max_gap = max(
        (float(buf[k]["start"]) - float(buf[k - 1]["end"]) for k in range(1, len(buf))),
        default=0.0,
    )
    if max_gap >= 0.45:
        return True
    return False


def partition_words_into_blocks(
    words: list[dict[str, Any]],
    *,
    hard_gap_s: float = 1.2,
    max_block_words: int = 80,
    max_cue_chars: float = 28.0,
    max_cue_duration_s: float = 5.5,
) -> tuple[list[dict[str, Any]], list[WordBlock]]:
    """Normalize word IDs (`1..N`) and partition words into acoustically contiguous `WordBlock`s."""
    cleaned = clamp_stretched_words(words)
    norm_words: list[dict[str, Any]] = []
    for raw in cleaned:
        txt = str(raw.get("text") or "").strip()
        if not txt:
            continue
        item = dict(raw)
        item["id"] = len(norm_words) + 1
        item["text"] = txt
        item["start"] = max(0.0, float(item.get("start", 0.0)))
        item["end"] = max(item["start"] + 0.05, float(item.get("end", item["start"] + 0.05)))
        norm_words.append(item)

    if not norm_words:
        return [], []

    raw_groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = [norm_words[0]]
    for w in norm_words[1:]:
        prev = cur[-1]
        gap = float(w["start"]) - float(prev["end"])
        prev_spk = prev.get("speaker")
        cur_spk = w.get("speaker")
        speaker_changed = bool(prev_spk and cur_spk and prev_spk != cur_spk)
        seg_jump = w.get("seg") != prev.get("seg") and gap >= 0.8
        if gap >= hard_gap_s or speaker_changed or seg_jump:
            raw_groups.append(cur)
            cur = [w]
        else:
            cur.append(w)
    raw_groups.append(cur)

    blocks: list[WordBlock] = []
    for grp in raw_groups:
        for sub in _split_oversized_group(grp, max_block_words=max_block_words):
            blocks.append(
                WordBlock(
                    block_id=len(blocks) + 1,
                    words=sub,
                    first_wid=int(sub[0]["id"]),
                    last_wid=int(sub[-1]["id"]),
                    needs_llm=_block_needs_llm(
                        sub,
                        max_cue_chars=max_cue_chars,
                        max_cue_duration_s=max_cue_duration_s,
                    ),
                )
            )
    return norm_words, blocks


def format_block_words_for_llm(
    words: list[dict[str, Any]],
    *,
    pause_hint_s: float = 0.35,
) -> str:
    """Format a word block as compact `<wid>:<word>` tokens with `(0.5s)` pause markers."""
    parts: list[str] = []
    for idx, w in enumerate(words):
        if idx > 0:
            gap = float(w["start"]) - float(words[idx - 1]["end"])
            if gap >= pause_hint_s:
                parts.append(f"({gap:.1f}s)")
        parts.append(f"{int(w['id'])}:{w['text']}")
    return " ".join(parts)


def _rule_split_block_spans(
    block: WordBlock,
    *,
    max_cue_chars: float = 34.0,
    max_cue_duration_s: float = 6.0,
) -> list[tuple[int, int]]:
    """Deterministic fallback splitter producing valid 1-based inclusive `(start_wid, end_wid)` spans."""
    words = block.words
    if not words:
        return []
    spans: list[tuple[int, int]] = []
    buf: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        spans.append((int(buf[0]["id"]), int(buf[-1]["id"])))
        buf = []

    for w in words:
        if buf:
            prev = buf[-1]
            gap = float(w["start"]) - float(prev["end"])
            cur_text = _join_word_buf(buf)
            cur_w = _effective_display_chars(cur_text)
            cur_dur = float(prev["end"]) - float(buf[0]["start"])
            next_w = cur_w + _effective_display_chars(str(w["text"])) + (0.5 if w.get("space_before", True) else 0.0)
            next_dur = float(w["end"]) - float(buf[0]["start"])
            if gap >= 0.8:
                flush()
            elif _SENT_END_WORD_RE.search(str(prev["text"])) and (len(buf) >= 3 or cur_dur >= 1.8 or cur_w >= 12.0):
                flush()
            elif _CLAUSE_PUNCT_WORD_RE.search(str(prev["text"])) and (cur_w >= max_cue_chars * 0.65 or cur_dur >= 3.8):
                flush()
            elif next_w > max_cue_chars or next_dur > max_cue_duration_s:
                flush()
        buf.append(w)
    flush()
    return spans


def _build_resegment_system_prompt(
    *,
    max_cue_chars: float = 28.0,
    max_cue_duration_s: float = 5.5,
) -> str:
    latin_chars = int(round(max_cue_chars * 2))
    cjk_chars = int(round(max_cue_chars))
    return (
        "You are a professional subtitle timing and segmentation editor.\n"
        "You receive JSON with `blocks`: each item has `block_id` and `words`, where every spoken word is "
        "prefixed with its integer ID (`<id>:<word>`) and notable internal pauses appear as `(0.5s)`.\n\n"
        "Task: Decide optimal subtitle cue boundaries inside each block so every resulting subtitle is a natural, "
        "self-contained semantic unit (complete clause, sentence, or short utterance) comfortable to read on screen.\n\n"
        "Rules:\n"
        "1. For EVERY input `block_id`, return a strictly ascending list of ending word IDs (`[end_id_1, end_id_2, ...]`) "
        "where the last element equals the block's final word ID. If a block is already one clean subtitle, return `[final_word_id]`.\n"
        f"2. Target length per cue: up to ~{cjk_chars} CJK characters OR ~{latin_chars} Latin/Cyrillic characters "
        f"(and <= {max_cue_duration_s:g}s). Split run-on blocks at natural sentence endings (`.!?`), clause commas, "
        "conjunctions, or `(0.5s)` pause markers.\n"
        "3. Keep tightly connected fragments together when they form one short thought (e.g., split words or a 1-2 word "
        "trailing fragment across a brief pause) — do NOT over-split into 1-2 word fragments unless it is an independent "
        "exclamation or complete short sentence (e.g. \"Alone.\", \"Опа!\", \"Без шуток?\").\n"
        "4. Never split inside a proper noun, hero/item/character name, quote prefix + short quote, or tight modifier+noun pair.\n"
        "5. Account for EVERY input `block_id`. Return ONLY compact JSON:\n"
        '{"splits": {"1": [19, 26], "2": [520, 527]}}'
    )


def _extract_raw_splits_map(parsed: dict[str, Any]) -> dict[int, list[int]]:
    """Extract `{block_id: [cut_wid, ...]}` tolerating minor wrapper key variations."""
    raw_map: dict[int, list[int]] = {}
    if not isinstance(parsed, dict):
        return raw_map

    def _to_int_list(val: Any) -> list[int]:
        if isinstance(val, (int, str)):
            try:
                return [int(val)]
            except ValueError:
                return []
        if not isinstance(val, (list, tuple)):
            return []
        out: list[int] = []
        for item in val:
            if isinstance(item, (list, tuple)) and item:
                # Tolerate `[start_wid, end_wid]` pairs by taking `end_wid`
                item = item[-1]
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out

    splits_obj = parsed.get("splits")
    if isinstance(splits_obj, dict):
        for k, v in splits_obj.items():
            try:
                bid = int(k)
            except (TypeError, ValueError):
                continue
            raw_map[bid] = _to_int_list(v)
        return raw_map

    blocks_list = parsed.get("blocks")
    if isinstance(blocks_list, list):
        for row in blocks_list:
            if not isinstance(row, dict):
                continue
            raw_bid = row.get("block_id", row.get("id"))
            if raw_bid is None:
                continue
            try:
                bid = int(raw_bid)
            except (TypeError, ValueError):
                continue
            cuts_val = row.get("cuts", row.get("splits", row.get("ends")))
            raw_map[bid] = _to_int_list(cuts_val)
        return raw_map

    # Direct numeric keys fallback: {"1": [19, 26], ...}
    for k, v in parsed.items():
        if isinstance(k, str) and k.isdigit():
            raw_map[int(k)] = _to_int_list(v)
    return raw_map


def _validate_block_splits(
    parsed: dict[str, Any],
    batch_blocks: list[WordBlock],
    *,
    hard_max_chars: float = 38.0,
    hard_max_duration_s: float = 6.8,
) -> tuple[dict[int, list[tuple[int, int]]], list[WordBlock], list[tuple[WordBlock, list[str]]]]:
    """Validate LLM word-anchor cut IDs for each block and collect missing/violating blocks."""
    raw_map = _extract_raw_splits_map(parsed)
    spans_by_block: dict[int, list[tuple[int, int]]] = {}
    missing_blocks: list[WordBlock] = []
    violating_blocks: list[tuple[WordBlock, list[str]]] = []

    for blk in batch_blocks:
        if blk.block_id not in raw_map:
            missing_blocks.append(blk)
            continue
        cuts = sorted({
            cid for cid in raw_map[blk.block_id]
            if blk.first_wid <= cid <= blk.last_wid
        })
        if not cuts or cuts[-1] != blk.last_wid:
            cuts.append(blk.last_wid)

        spans: list[tuple[int, int]] = []
        cursor = blk.first_wid
        for end_wid in cuts:
            if end_wid >= cursor:
                spans.append((cursor, end_wid))
                cursor = end_wid + 1

        by_wid = {int(w["id"]): idx for idx, w in enumerate(blk.words)}
        issues: list[str] = []
        for s_wid, e_wid in spans:
            si = by_wid.get(s_wid)
            ei = by_wid.get(e_wid)
            if si is None or ei is None:
                continue
            sub = blk.words[si : ei + 1]
            if len(sub) <= 2:
                continue
            eff_chars = _effective_display_chars(_join_word_buf(sub))
            dur = float(sub[-1]["end"]) - float(sub[0]["start"])
            if eff_chars > hard_max_chars or dur > hard_max_duration_s:
                issues.append(
                    f"span [{s_wid}..{e_wid}] has {eff_chars:.1f} display chars / {dur:.1f}s "
                    f"(limit {hard_max_chars:.0f} chars / {hard_max_duration_s:.1f}s)"
                )

        spans_by_block[blk.block_id] = spans
        if issues:
            violating_blocks.append((blk, issues))

    return spans_by_block, missing_blocks, violating_blocks


def _batch_candidate_blocks(
    candidates: list[WordBlock],
    *,
    max_words_per_call: int = 900,
    max_blocks_per_call: int = 80,
) -> list[list[WordBlock]]:
    if not candidates:
        return []
    batches: list[list[WordBlock]] = []
    cur: list[WordBlock] = []
    cur_words = 0
    for blk in candidates:
        b_len = len(blk.words)
        if cur and (cur_words + b_len > max_words_per_call or len(cur) >= max_blocks_per_call):
            batches.append(cur)
            cur = []
            cur_words = 0
        cur.append(blk)
        cur_words += b_len
    if cur:
        batches.append(cur)
    return batches


def _resegment_cache_key(
    spec: ModelSpec,
    *,
    system_prompt: str,
    user_prompt: str,
    hard_max_chars: float,
    hard_max_duration_s: float,
) -> str:
    return make_cache_key(
        "resegment",
        {
            "v": "resegment_v1",
            "spec": {
                "provider": spec.provider,
                "model": spec.model,
                "reasoning_effort": spec.reasoning_effort,
                "thinking_level": spec.thinking_level,
                "temperature": spec.temperature,
            },
            "system_sha256": sha256_text(system_prompt),
            "user_sha256": sha256_text(user_prompt),
            "hard_max_chars": round(float(hard_max_chars), 2),
            "hard_max_duration_s": round(float(hard_max_duration_s), 2),
        },
    )


def resegment_words_with_llm(
    words: list[dict[str, Any]],
    *,
    translator: Any,
    hard_gap_s: float = 1.2,
    max_block_words: int = 80,
    max_cue_chars: float = 28.0,
    max_cue_duration_s: float = 5.5,
    max_words_per_call: int = 900,
    max_blocks_per_call: int = 80,
    repair_violations: bool = True,
    progress_fn: Callable[[str], None] | None = None,
) -> tuple[list[SrtCue], dict[str, Any]]:
    """Semantically re-segment word-level timestamps using Word-Anchor LLM cut IDs.

    Shares `translator`'s multi-provider fallback chain, SQLite cache (`namespace="resegment"`),
    and JSONL debug logger (`phase="resegment"`). Always returns a complete, drift-free `list[SrtCue]`
    (falling back to deterministic rule splitting for any block that fails LLM processing).
    """
    emit = progress_fn or getattr(translator, "_emit", None) or (lambda _m: None)
    norm_words, blocks = partition_words_into_blocks(
        words,
        hard_gap_s=hard_gap_s,
        max_block_words=max_block_words,
        max_cue_chars=max_cue_chars,
        max_cue_duration_s=max_cue_duration_s,
    )
    if not norm_words:
        return [], {"total_words": 0, "total_blocks": 0, "candidate_blocks": 0, "cues": 0, "cache_hits": 0}

    candidates = [b for b in blocks if b.needs_llm]
    spans_by_block: dict[int, list[tuple[int, int]]] = {
        b.block_id: [(b.first_wid, b.last_wid)]
        for b in blocks
        if not b.needs_llm
    }

    hard_max_chars = max(max_cue_chars + 10.0, max_cue_chars * 1.35)
    hard_max_duration_s = max(max_cue_duration_s + 1.3, 6.8)
    system_prompt = _build_resegment_system_prompt(
        max_cue_chars=max_cue_chars,
        max_cue_duration_s=max_cue_duration_s,
    )
    batches = _batch_candidate_blocks(
        candidates,
        max_words_per_call=max_words_per_call,
        max_blocks_per_call=max_blocks_per_call,
    )
    cache: ResultCache | None = getattr(translator, "cache", None)
    spec_chain: list[ModelSpec] = list(getattr(translator, "spec_chain", []) or [])
    cache_hits = 0
    repaired_blocks_total = 0
    rule_fallback_blocks = 0

    emit(
        f"  [词级重断句] 共 {len(norm_words)} 词 / {len(blocks)} 语音段"
        f"（其中 {len(candidates)} 段需语义断句，分为 {len(batches)} 批）"
    )

    for b_idx, batch in enumerate(batches, 1):
        label = f"重断句 {b_idx}/{len(batches)}"
        payload_obj = {
            "blocks": [
                {"block_id": blk.block_id, "words": format_block_words_for_llm(blk.words)}
                for blk in batch
            ]
        }
        user_prompt = json.dumps(payload_obj, ensure_ascii=False)

        hit_used = False
        if cache is not None and spec_chain:
            primary_key = _resegment_cache_key(
                spec_chain[0],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                hard_max_chars=hard_max_chars,
                hard_max_duration_s=hard_max_duration_s,
            )
            cached = cache.get("resegment", primary_key)
            if cached is not None and isinstance(cached.get("spans"), dict):
                raw_cached = cached["spans"]
                ok = True
                parsed_cached: dict[int, list[tuple[int, int]]] = {}
                for blk in batch:
                    item = raw_cached.get(str(blk.block_id))
                    if not isinstance(item, list) or not item:
                        ok = False
                        break
                    parsed_cached[blk.block_id] = [(int(p[0]), int(p[1])) for p in item]
                if ok:
                    spans_by_block.update(parsed_cached)
                    cache_hits += 1
                    hit_used = True
                    translator.stats.append({
                        "phase": "resegment",
                        "provider": str(cached.get("provider") or spec_chain[0].provider),
                        "model": str(cached.get("model") or spec_chain[0].model),
                        "service_tier": str(cached.get("service_tier") or spec_chain[0].service_tier),
                        "cues": sum(len(v) for v in parsed_cached.values()),
                        "kept": 0,
                        "dropped": 0,
                        "elapsed_s": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_tokens": 0,
                        "cache_hit": True,
                    })
                    emit(
                        f"  [{label}] 命中本地缓存 ({cached.get('provider')}/{cached.get('model')}) | "
                        f"{len(batch)} 段 -> {sum(len(v) for v in parsed_cached.values())} 条 [cache]"
                    )
        if hit_used:
            continue

        try:
            res, spec, elapsed = translator._call_json_with_fallback(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                task_label=label,
                action_desc=f"语义重断句 {len(batch)} 段（{sum(len(b.words) for b in batch)} 词）",
                phase="resegment",
            )
        except Exception as exc:
            emit(f"  [{label}] LLM 重断句失败，本批降级为确定性规则断句: {exc}")
            for blk in batch:
                spans_by_block[blk.block_id] = _rule_split_block_spans(
                    blk, max_cue_chars=hard_max_chars, max_cue_duration_s=max_cue_duration_s,
                )
                rule_fallback_blocks += 1
            continue

        batch_spans, missing_blks, violating_blks = _validate_block_splits(
            res.parsed,
            batch,
            hard_max_chars=hard_max_chars,
            hard_max_duration_s=hard_max_duration_s,
        )
        used_tier = res.service_tier or spec.service_tier
        translator.stats.append({
            "phase": "resegment",
            "provider": spec.provider,
            "model": spec.model,
            "service_tier": used_tier,
            "cues": sum(len(v) for v in batch_spans.values()),
            "kept": 0,
            "dropped": 0,
            "elapsed_s": elapsed,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "cached_tokens": res.cached_tokens,
        })

        need_repair_blks: list[WordBlock] = list(missing_blks)
        diagnostics: dict[str, str] = {
            str(blk.block_id): "missing from previous response — return ending word IDs for this block"
            for blk in missing_blks
        }
        for blk, issues in violating_blks:
            if all(existing.block_id != blk.block_id for existing in need_repair_blks):
                need_repair_blks.append(blk)
            diagnostics[str(blk.block_id)] = "; ".join(issues)

        unresolved_after_repair = len(need_repair_blks)
        if repair_violations and need_repair_blks:
            emit(
                f"  [{label}] 检测到 {len(need_repair_blks)} 段漏切/超长，启动 1 轮精确自修复..."
            )
            repair_payload = {
                "repair_instruction": (
                    "Fix ONLY the listed blocks where your previous splits were missing or left an overlong span. "
                    "Insert additional natural cut word IDs so every span stays concise. "
                    'Return ONLY compact JSON: {"splits": {"<block_id>": [end_id_1, ...]}}'
                ),
                "issues": diagnostics,
                "blocks": [
                    {"block_id": blk.block_id, "words": format_block_words_for_llm(blk.words)}
                    for blk in need_repair_blks
                ],
            }
            try:
                rep_res, rep_spec, rep_elapsed = translator._call_json_with_fallback(
                    system_prompt=system_prompt,
                    user_prompt=json.dumps(repair_payload, ensure_ascii=False),
                    task_label=f"{label} 修复",
                    action_desc=f"修复 {len(need_repair_blks)} 段断句",
                    phase="resegment_repair",
                    preferred_spec=spec,
                )
                rep_spans, rep_missing, rep_violating = _validate_block_splits(
                    rep_res.parsed,
                    need_repair_blks,
                    hard_max_chars=hard_max_chars,
                    hard_max_duration_s=hard_max_duration_s,
                )
                rep_bad_ids = {b.block_id for b in rep_missing} | {b.block_id for b, _ in rep_violating}
                for bid, sp in rep_spans.items():
                    if bid not in rep_bad_ids or bid not in batch_spans:
                        batch_spans[bid] = sp
                        if bid not in rep_bad_ids:
                            repaired_blocks_total += 1
                unresolved_after_repair = len(rep_bad_ids)
                translator.stats.append({
                    "phase": "resegment_repair",
                    "provider": rep_spec.provider,
                    "model": rep_spec.model,
                    "service_tier": rep_res.service_tier or rep_spec.service_tier,
                    "cues": 0,
                    "kept": 0,
                    "dropped": 0,
                    "elapsed_s": rep_elapsed,
                    "input_tokens": rep_res.input_tokens,
                    "output_tokens": rep_res.output_tokens,
                    "cached_tokens": rep_res.cached_tokens,
                })
                emit(
                    f"  [{label}] 自修复完成 ({rep_spec.provider}/{rep_spec.model}) | 耗时 {rep_elapsed:.1f}s | "
                    f"剩余异常 {unresolved_after_repair} 段"
                )
            except Exception as rep_exc:
                emit(f"  [{label}] 自修复失败，对异常段使用规则兜底: {rep_exc}")

        # Final deterministic rule fallback for any block still missing or wildly overlong
        still_bad_ids: set[int] = set()
        for blk in batch:
            if blk.block_id not in batch_spans:
                still_bad_ids.add(blk.block_id)
            else:
                # Re-check extreme violation after repair
                by_wid = {int(w["id"]): idx for idx, w in enumerate(blk.words)}
                for s_wid, e_wid in batch_spans[blk.block_id]:
                    sub = blk.words[by_wid[s_wid] : by_wid[e_wid] + 1]
                    if len(sub) > 3 and (
                        _effective_display_chars(_join_word_buf(sub)) > hard_max_chars * 1.15
                        or (float(sub[-1]["end"]) - float(sub[0]["start"])) > hard_max_duration_s * 1.15
                    ):
                        still_bad_ids.add(blk.block_id)
                        break

        for bid in still_bad_ids:
            blk = next(b for b in batch if b.block_id == bid)
            batch_spans[bid] = _rule_split_block_spans(
                blk, max_cue_chars=hard_max_chars, max_cue_duration_s=max_cue_duration_s,
            )
            rule_fallback_blocks += 1

        spans_by_block.update(batch_spans)

        if cache is not None and not still_bad_ids:
            used_key = _resegment_cache_key(
                spec,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                hard_max_chars=hard_max_chars,
                hard_max_duration_s=hard_max_duration_s,
            )
            cache.set(
                "resegment",
                used_key,
                {
                    "provider": spec.provider,
                    "model": spec.model,
                    "service_tier": used_tier,
                    "spans": {str(blk.block_id): [list(p) for p in batch_spans[blk.block_id]] for blk in batch},
                },
                meta={"provider": spec.provider, "model": spec.model, "blocks": len(batch)},
            )

        out_cues_n = sum(len(batch_spans[blk.block_id]) for blk in batch)
        cache_part = f" ({res.cached_tokens} cached)" if res.cached_tokens else ""
        emit(
            f"  [{label}] 成功 ({spec.provider}/{spec.model}:{used_tier}) | 耗时 {elapsed:.1f}s | "
            f"{len(batch)} 段 -> {out_cues_n} 条 | Token: {res.input_tokens} in{cache_part} / {res.output_tokens} out"
        )

    all_spans: list[tuple[int, int]] = []
    for blk in blocks:
        all_spans.extend(spans_by_block[blk.block_id])

    cues = reconstruct_cues_from_word_spans(norm_words, all_spans)
    summary = {
        "total_words": len(norm_words),
        "total_blocks": len(blocks),
        "candidate_blocks": len(candidates),
        "cues": len(cues),
        "cache_hits": cache_hits,
        "repaired_blocks": repaired_blocks_total,
        "rule_fallback_blocks": rule_fallback_blocks,
    }
    return cues, summary


def resegment_file(
    input_path: Path,
    *,
    defaults_path: Path = DEFAULT_CONFIG_PATH,
    profile_name: str | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
    max_cue_chars_override: float | None = None,
    max_cue_duration_override: float | None = None,
    use_cache_override: bool | None = None,
    log_llm_override: bool | None = None,
    output_path: Path | None = None,
) -> Path:
    """Standalone CLI helper: semantically re-segment `.words.json` (or `.srt` with sidecar) into `.srt`."""
    from src.translate import SubtitleTranslator, _resolve_translate_models_and_providers

    words_path = find_sidecar_words_json(input_path)
    if words_path is None:
        raise FileNotFoundError(
            f"未找到词级时间戳文件（请传入 .words.json，或确保同目录存在 {input_path.stem}.words.json）"
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

    words = load_words_json(words_path)
    if not words:
        raise ValueError(f"词级时间戳文件为空: {words_path}")

    max_cue_chars = float(
        max_cue_chars_override
        if max_cue_chars_override is not None
        else profile.translate.get("resegment_max_cue_chars", tl_cfg.get("resegment_max_cue_chars", 28))
    )
    max_cue_dur = float(
        max_cue_duration_override
        if max_cue_duration_override is not None
        else profile.translate.get("resegment_max_cue_duration_s", tl_cfg.get("resegment_max_cue_duration_s", 5.5))
    )
    hard_gap_s = float(
        profile.translate.get("resegment_hard_gap_s", tl_cfg.get("resegment_hard_gap_s", 1.2))
    )
    retry_503 = int(tl_cfg.get("retry_503", 3))
    cache = build_result_cache(defaults, use_cache_override=use_cache_override)
    llm_logger = build_llm_logger(defaults, enabled_override=log_llm_override)

    stem = words_path.name[: -len(".words.json")] if words_path.name.lower().endswith(".words.json") else words_path.stem
    out = output_path or words_path.with_name(f"{stem}_resegmented.srt")

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

    print("=== [AutoSub 词级语义重断句] ===", flush=True)
    print(f"输入词表: {words_path}（共 {len(words)} 词）", flush=True)
    print(f"输出字幕: {out}", flush=True)
    print(f"模型链: {' -> '.join(s.display_label for s in active_specs)}", flush=True)

    cues, info = resegment_words_with_llm(
        words,
        translator=translator,
        hard_gap_s=hard_gap_s,
        max_cue_chars=max_cue_chars,
        max_cue_duration_s=max_cue_dur,
        progress_fn=lambda msg: print(msg, flush=True),
    )
    write_srt([(c.start, c.end, [c.text]) for c in cues], out)
    total_s = sum(float(s.get("elapsed_s", 0.0)) for s in translator.stats)
    total_in = sum(int(s.get("input_tokens", 0)) for s in translator.stats)
    total_out = sum(int(s.get("output_tokens", 0)) for s in translator.stats)
    print(
        f"[重断句完成] {out} | {info['total_words']} 词 -> {len(cues)} 条语义字幕 | "
        f"耗时 {total_s:.1f}s | Token: {total_in} in / {total_out} out",
        flush=True,
    )
    return out


def run_resegment_cli(
    input_path: Path,
    *,
    defaults_path: Path = DEFAULT_CONFIG_PATH,
    profile_name: str | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    reasoning_effort_override: str | None = None,
    max_cue_chars_override: float | None = None,
    max_cue_duration_override: float | None = None,
    use_cache_override: bool | None = None,
    log_llm_override: bool | None = None,
    output_path: Path | None = None,
) -> int:
    try:
        resegment_file(
            input_path,
            defaults_path=defaults_path,
            profile_name=profile_name,
            provider_override=provider_override,
            model_override=model_override,
            reasoning_effort_override=reasoning_effort_override,
            max_cue_chars_override=max_cue_chars_override,
            max_cue_duration_override=max_cue_duration_override,
            use_cache_override=use_cache_override,
            log_llm_override=log_llm_override,
            output_path=output_path,
        )
        return 0
    except Exception as exc:
        print(f"[错误] {exc}", flush=True)
        return 1
