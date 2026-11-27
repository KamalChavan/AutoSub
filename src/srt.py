"""SRT parsing, incremental writing, timestamp remapping, and word-to-cue grouping."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SrtCue:
    id: int
    start: float
    end: float
    text: str

    @property
    def ts_line(self) -> str:
        return f"{format_srt_ts(self.start)} --> {format_srt_ts(self.end)}"


def format_srt_ts(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt_ts(ts: str) -> float:
    h, m, s, ms = map(int, re.split(r"[:,]", ts.strip()))
    return h * 3600 + m * 60 + s + ms / 1000.0


def parse_srt(path: Path) -> list[SrtCue]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    blocks = re.split(r"\r?\n\s*\r?\n", raw)
    cues: list[SrtCue] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        try:
            idx = int(lines[0].lstrip("\ufeff"))
        except ValueError:
            continue
        a, b = [part.strip() for part in lines[1].split("-->", 1)]
        cues.append(
            SrtCue(
                id=idx,
                start=parse_srt_ts(a),
                end=parse_srt_ts(b),
                text=" ".join(lines[2:]),
            )
        )
    return cues


def write_srt(cues: list[tuple[float, float, list[str]]], path: Path) -> None:
    lines: list[str] = []
    for idx, (start, end, text_lines) in enumerate(cues, 1):
        lines.append(str(idx))
        lines.append(f"{format_srt_ts(start)} --> {format_srt_ts(end)}")
        lines.extend(text_lines)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_srt_cues(
    cues: list[tuple[float, float, list[str]]],
    path: Path,
    *,
    start_index: int = 1,
    truncate: bool = False,
) -> int:
    """Incrementally append cues to an .srt file and flush immediately; returns cumulative count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if truncate else "a"
    idx = start_index - 1
    with path.open(mode, encoding="utf-8") as fh:
        for start, end, text_lines in cues:
            idx += 1
            fh.write(f"{idx}\n{format_srt_ts(start)} --> {format_srt_ts(end)}\n")
            for ln in text_lines:
                fh.write(f"{ln}\n")
            fh.write("\n")
        fh.flush()
    return idx


def map_word_to_original(
    w_start: float,
    w_end: float,
    seg_map: list[dict[str, float | int]],
) -> tuple[int, float, float]:
    """Map compact VAD-chunk word timestamps back to the original media timeline."""
    mid = 0.5 * (w_start + w_end)
    best = seg_map[0]
    best_dist = float("inf")
    for seg in seg_map:
        cs, ce = float(seg["compact_start"]), float(seg["compact_end"])
        if cs <= mid <= ce:
            best = seg
            break
        d = min(abs(mid - cs), abs(mid - ce))
        if d < best_dist:
            best_dist = d
            best = seg
    os_ = float(best["orig_start"]) + max(0.0, w_start - float(best["compact_start"]))
    oe_ = float(best["orig_start"]) + max(0.0, w_end - float(best["compact_start"]))
    orig_end = float(best["orig_end"])
    os_ = min(orig_end, os_)
    oe_ = min(orig_end, max(os_ + 0.05, oe_))
    return int(best["idx"]), os_, oe_


def clamp_stretched_words(
    words: list[dict[str, Any]],
    *,
    min_cap_s: float = 1.8,
) -> list[dict[str, Any]]:
    """Clamp anomalous single-word acoustic timestamp stretches (e.g. 4s-14s trailing/leading silence)."""
    if not words:
        return []
    out: list[dict[str, Any]] = []
    n = len(words)
    for i, raw in enumerate(words):
        w = dict(raw)
        ws = float(w["start"])
        we = float(w["end"])
        txt = str(w.get("text") or "").strip()
        max_dur = max(min_cap_s, 0.12 * len(txt) + 0.9)
        if (we - ws) > max_dur:
            anchored_to_next = False
            if i + 1 < n:
                nxt = words[i + 1]
                if nxt.get("seg") == w.get("seg") and (float(nxt["start"]) - we) <= 0.5:
                    anchored_to_next = True
            if anchored_to_next:
                prev_end = float(out[-1]["end"]) if out and out[-1].get("seg") == w.get("seg") else 0.0
                ws = max(prev_end, we - max_dur)
            else:
                we = ws + max_dur
            w["start"] = ws
            w["end"] = max(ws + 0.05, we)
        out.append(w)
    return out


def _join_word_buf(buf: list[dict[str, Any]]) -> str:
    if not buf:
        return ""
    parts = [str(buf[0]["text"])]
    for w in buf[1:]:
        sep = " " if w.get("space_before", True) else ""
        parts.append(sep + str(w["text"]))
    return "".join(parts)


def _should_absorb_orphan_tail(
    words: list[dict[str, Any]],
    idx: int,
    buf: list[dict[str, Any]],
    cur_text: str,
) -> bool:
    """Avoid soft-splitting when only 1-2 short trailing words remain before a sentence/pause boundary."""
    n = len(words)
    tail_words: list[dict[str, Any]] = []
    prev = buf[-1]
    prev_spk = prev.get("speaker")
    for j in range(idx, min(n, idx + 3)):
        cand = words[j]
        gap = float(cand["start"]) - float(prev["end"])
        cand_spk = cand.get("speaker")
        spk_changed = bool(prev_spk and cand_spk and prev_spk != cand_spk)
        if cand.get("seg") != prev.get("seg") or spk_changed or gap >= 0.45:
            break
        tail_words.append(cand)
        prev = cand
        prev_spk = cand_spk
        if re.search(r"[.!?…。！？]$", str(cand.get("text") or "")):
            break
    else:
        # More than 2 continuous words follow without closing punctuation; not an orphan tail
        if idx + 2 < n:
            after = words[idx + 2]
            if (
                after.get("seg") == prev.get("seg")
                and (float(after["start"]) - float(prev["end"])) < 0.45
                and not re.search(r"[.!?…。！？]$", str(prev.get("text") or ""))
            ):
                return False

    if not (1 <= len(tail_words) <= 2):
        return False
    tail_text = _join_word_buf(tail_words)
    total_dur = float(tail_words[-1]["end"]) - float(buf[0]["start"])
    return len(tail_text) <= 18 and (len(cur_text) + 1 + len(tail_text)) <= 92 and total_dur <= 7.2


def group_words_to_cues(words: list[dict[str, Any]]) -> list[SrtCue]:
    """Group word-level timestamps into readable SRT cues without crossing VAD segment or speaker boundaries."""
    if not words:
        return []
    cleaned_words = clamp_stretched_words(words)
    cues: list[SrtCue] = []
    buf: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        start = float(buf[0]["start"])
        end = float(buf[-1]["end"])
        if cues and start < cues[-1].end:
            start = cues[-1].end
        if end <= start:
            end = start + 0.15
        cues.append(SrtCue(id=len(cues) + 1, start=start, end=end, text=_join_word_buf(buf)))
        buf = []

    for idx, w in enumerate(cleaned_words):
        if buf:
            prev = buf[-1]
            gap = float(w["start"]) - float(prev["end"])
            cur_text = _join_word_buf(buf)
            cur_dur = float(prev["end"]) - float(buf[0]["start"])
            prev_spk = prev.get("speaker")
            cur_spk = w.get("speaker")
            speaker_changed = bool(prev_spk and cur_spk and prev_spk != cur_spk)
            next_sep_len = 1 if w.get("space_before", True) else 0
            if w.get("seg") != prev.get("seg") or speaker_changed or gap >= 0.8:
                flush()
            elif re.search(r"[.!?…。！？]$", prev["text"]) and (
                len(buf) >= 3 or cur_dur >= 1.8 or len(cur_text) >= 24
            ):
                flush()
            elif re.search(r"[,;:—，；：]$", prev["text"]) and (len(cur_text) >= 48 or cur_dur >= 4.0):
                if not _should_absorb_orphan_tail(cleaned_words, idx, buf, cur_text):
                    flush()
            elif len(cur_text) + next_sep_len + len(w["text"]) > 76 or (float(w["end"]) - float(buf[0]["start"])) > 6.0:
                if not _should_absorb_orphan_tail(cleaned_words, idx, buf, cur_text):
                    flush()
        buf.append(w)
    flush()
    return cues


def _effective_display_chars(text: str) -> float:
    """Count display reading units (CJK/wide chars=1.0, ASCII chars=0.5)."""
    return sum(1.0 if ord(ch) > 127 else 0.5 for ch in (text or "").strip())


def polish_srt_timing(
    cues: list[tuple[float, float, list[str]]],
    *,
    min_duration: float = 1.0,
    max_cps: float = 13.0,
    max_linger: float = 1.2,
    min_gap: float = 0.06,
    bridge_gap: float = 0.15,
    lead_in: float = 0.05,
) -> list[tuple[float, float, list[str]]]:
    """Deterministic subtitle timing polish: gentle lead-in, CPS/min-duration linger, and micro-gap bridging.

    Never overlaps adjacent cues and never shifts start times late.
    """
    if not cues:
        return []

    items = [[max(0.0, float(s)), max(float(s) + 0.05, float(e)), list(lines)] for s, e, lines in cues]
    n = len(items)

    # 1. Gentle lead-in so subtitles appear ~50ms ahead of speech onset without colliding with previous cue
    if lead_in > 0:
        for i in range(n):
            floor = (items[i - 1][1] + min_gap) if i > 0 else 0.0
            items[i][0] = max(floor, items[i][0] - lead_in)

    # 2. Extend ultra-short / high-CPS cues into following silence (bounded by max_linger & next cue start)
    for i in range(n):
        start, end, lines = items[i]
        primary_line = lines[0] if lines else ""
        eff_chars = _effective_display_chars(primary_line)
        desired_dur = max(min_duration, eff_chars / max(1.0, max_cps))
        if (end - start) < desired_dur:
            target_end = min(start + desired_dur, end + max_linger)
            if i + 1 < n:
                target_end = min(target_end, items[i + 1][0] - min_gap)
            if target_end > end:
                items[i][1] = target_end

    # 3. Close tiny visual flicker gaps (< bridge_gap) seamlessly
    if bridge_gap > 0:
        for i in range(n - 1):
            gap = items[i + 1][0] - items[i][1]
            if 0.0 < gap <= bridge_gap:
                items[i][1] = items[i + 1][0]

    # 4. Enforce strict monotonic non-overlapping invariant
    result: list[tuple[float, float, list[str]]] = []
    for i, (start, end, lines) in enumerate(items):
        if result and start < result[-1][1]:
            start = result[-1][1]
        if end <= start:
            end = start + 0.05
        result.append((round(start, 3), round(end, 3), lines))
    return result


def _norm_compare_text(text: str) -> str:
    """Normalize punctuation, case, and trivial Russian spelling variants for dual-ASR diffing."""
    t = (text or "").lower().replace("ё", "е").replace("ъ", "ь")
    t = re.sub(r"\bблять\b", "блядь", t)
    t = re.sub(r"\bче\b", "что", t)
    cleaned = re.sub(r"[^\w]+", "", t)
    # Collapse doubled letters (e.g. спелл/спел, килл/кил, баттерфляй/батерфляй) for comparison only
    return re.sub(r"(.)\1+", r"\1", cleaned)


def align_reference_srt(
    primary_cues: list[SrtCue],
    ref_cues: list[SrtCue],
    *,
    pad: float = 0.5,
    max_similarity: float = 0.95,
) -> dict[int, str]:
    """Align a secondary ASR SRT track onto primary cue IDs for dual-ASR disambiguation.

    Returns `{primary_cue.id: ref_text}` only when overlapping reference text has a meaningful
    lexical difference from primary text (filtering out pure spelling/ё/filler noise).
    """
    if not primary_cues or not ref_cues:
        return {}

    from difflib import SequenceMatcher

    aligned: dict[int, str] = {}
    ref_idx = 0
    n_ref = len(ref_cues)

    for pc in primary_cues:
        win_start = pc.start - pad
        win_end = pc.end + pad
        while ref_idx < n_ref and ref_cues[ref_idx].end < win_start:
            ref_idx += 1

        strict_parts: list[str] = []
        prox_parts: list[str] = []
        j = ref_idx
        while j < n_ref and ref_cues[j].start <= win_end:
            rc = ref_cues[j]
            txt = rc.text.strip()
            if txt:
                overlap = min(pc.end, rc.end) - max(pc.start, rc.start)
                if overlap > 0.08:
                    strict_parts.append(txt)
                else:
                    rc_mid = 0.5 * (rc.start + rc.end)
                    pc_mid = 0.5 * (pc.start + pc.end)
                    if (win_start <= rc_mid <= win_end) or (rc.start - pad <= pc_mid <= rc.end + pad):
                        prox_parts.append(txt)
            j += 1

        parts = strict_parts if strict_parts else prox_parts
        if parts:
            ref_text = " ".join(parts).strip()
            if not ref_text:
                continue
            np_txt = _norm_compare_text(pc.text)
            nr_txt = _norm_compare_text(ref_text)
            if not nr_txt or np_txt == nr_txt:
                continue
            if max_similarity < 1.0 and SequenceMatcher(None, np_txt, nr_txt).ratio() >= max_similarity:
                continue
            aligned[pc.id] = ref_text

    return aligned
