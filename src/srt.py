"""SRT parsing, incremental writing, timestamp remapping, and word-to-cue grouping."""
from __future__ import annotations

import json
import math
import re
import unicodedata
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


_CLOSING_PUNCT = frozenset("，。！？；：、…,.!?;:)]}》】」』”’—–~～")
_OPENING_PUNCT = frozenset("([{《【「『“‘")
_STRONG_BREAK_PUNCT = frozenset("。！？!?；;…")
_MEDIUM_BREAK_PUNCT = frozenset("，,、：:—–")
_WRAP_TOKEN_RE = re.compile(
    r"\s+|[A-Za-z0-9\u00C0-\u024F\u0400-\u04FF]+(?:['’\-./:][A-Za-z0-9\u00C0-\u024F\u0400-\u04FF]+)*%?|."
)


def _char_display_width(ch: str) -> float:
    """Visual display width in CJK-equivalent units (CJK/fullwidth=1.0, Latin/Cyrillic/ASCII=0.5)."""
    if ch in ("\n", "\r"):
        return 0.0
    return 1.0 if unicodedata.east_asian_width(ch) in ("W", "F") else 0.5


def _effective_display_chars(text: str) -> float:
    """Count display reading units (CJK/wide chars=1.0, narrow Latin/Cyrillic/ASCII chars=0.5)."""
    return sum(_char_display_width(ch) for ch in (text or "").strip())


def _tokenize_for_wrap(text: str) -> list[tuple[str, bool, float]]:
    """Tokenize a single subtitle line into indivisible `(token_text, space_before, display_width)` units."""
    raw_items: list[tuple[str, bool]] = []
    pending_space = False
    for match in _WRAP_TOKEN_RE.finditer(text):
        piece = match.group(0)
        if piece.isspace():
            pending_space = True
            continue
        raw_items.append((piece, pending_space and bool(raw_items)))
        pending_space = False

    # Attach opening punctuation to the following token so it never dangles at line end
    with_open: list[tuple[str, bool]] = []
    pending_open = ""
    pending_open_sp = False
    for tok, sp in raw_items:
        if len(tok) == 1 and tok in _OPENING_PUNCT:
            if not pending_open:
                pending_open_sp = sp
            pending_open += tok
            continue
        if pending_open:
            with_open.append((pending_open + tok, pending_open_sp))
            pending_open = ""
            pending_open_sp = False
        else:
            with_open.append((tok, sp))
    if pending_open:
        with_open.append((pending_open, pending_open_sp))

    # Attach closing punctuation to the previous token so it never starts a new line
    merged: list[tuple[str, bool]] = []
    for tok, sp in with_open:
        if len(tok) == 1 and tok in _CLOSING_PUNCT and merged:
            prev_tok, prev_sp = merged[-1]
            merged[-1] = (prev_tok + tok, prev_sp)
        else:
            merged.append((tok, sp))

    return [(tok, sp, _effective_display_chars(tok)) for tok, sp in merged]


def _wrap_single_line(text: str, max_line_chars: float) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned or max_line_chars <= 0:
        return cleaned
    total_w = _effective_display_chars(cleaned)
    if total_w <= max_line_chars:
        return cleaned

    tokens = _tokenize_for_wrap(cleaned)
    n = len(tokens)
    if n <= 1:
        return cleaned

    pref_w = [0.0] * (n + 1)
    pref_sp = [0.0] * (n + 1)
    for idx, (_tok, sp, w) in enumerate(tokens):
        pref_w[idx + 1] = pref_w[idx] + w
        pref_sp[idx + 1] = pref_sp[idx] + (0.5 if sp else 0.0)

    def span_width(i: int, j: int) -> float:
        return (pref_w[j] - pref_w[i]) + (pref_sp[j] - pref_sp[i + 1])

    def break_bonus(end_j: int) -> float:
        if end_j >= n:
            return 0.0
        tok_text = tokens[end_j - 1][0]
        tail = tok_text[-2:] if len(tok_text) >= 2 else tok_text
        if any(ch in _STRONG_BREAK_PUNCT for ch in tail):
            return -75.0
        if any(ch in _MEDIUM_BREAK_PUNCT for ch in tail):
            return -55.0
        if tokens[end_j][1]:
            return -3.0
        return 0.0

    def solve_for_k(k: int) -> tuple[float, float, list[str]] | None:
        if k > n:
            return None
        target_w = total_w / k
        inf = float("inf")
        dp = [[inf] * (n + 1) for _ in range(k + 1)]
        parent = [[-1] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 0.0

        soft_cap = max_line_chars * 1.18
        for line_idx in range(1, k + 1):
            min_i = line_idx - 1
            max_i = n - (k - line_idx + 1)
            for i in range(min_i, max_i + 1):
                base_cost = dp[line_idx - 1][i]
                if base_cost == inf:
                    continue
                max_j = n - (k - line_idx)
                for j in range(i + 1, max_j + 1):
                    w = span_width(i, j)
                    if j > i + 1 and w > soft_cap and line_idx < k:
                        break
                    overflow = max(0.0, w - max_line_chars)
                    overflow_pen = (50.0 + 40.0 * overflow * overflow) if overflow > 0 else 0.0
                    cost = base_cost + (w - target_w) ** 2 + overflow_pen + break_bonus(j)
                    if cost < dp[line_idx][j]:
                        dp[line_idx][j] = cost
                        parent[line_idx][j] = i

        if dp[k][n] == inf:
            return None
        ranges: list[tuple[int, int]] = []
        cur = n
        for line_idx in range(k, 0, -1):
            prev = parent[line_idx][cur]
            if prev < 0:
                return None
            ranges.append((prev, cur))
            cur = prev
        ranges.reverse()

        lines: list[str] = []
        max_overflow = 0.0
        for i, j in ranges:
            w = span_width(i, j)
            max_overflow = max(max_overflow, max(0.0, w - max_line_chars))
            parts = [tokens[i][0]]
            for m in range(i + 1, j):
                parts.append((" " if tokens[m][1] else "") + tokens[m][0])
            lines.append("".join(parts))
        return dp[k][n], max_overflow, lines

    min_k = max(2, math.ceil(total_w / max_line_chars))
    best = solve_for_k(min_k)
    if best is None or best[1] > 1.5:
        alt = solve_for_k(min_k + 1)
        if alt is not None and (best is None or alt[1] + 0.5 < best[1]):
            best = alt

    if best is None:
        return cleaned
    return "\n".join(best[2])


def wrap_subtitle_line(text: str, *, max_line_chars: float = 24.0) -> str:
    """Balance-wrap a subtitle string when its visual display width exceeds `max_line_chars` CJK units.

    Set `max_line_chars <= 0` to disable wrapping. Preserves word boundaries and avoids dangling
    opening/closing punctuation while keeping wrapped lines visually balanced.
    """
    raw = (text or "").strip()
    if not raw or max_line_chars <= 0:
        return raw
    if "\n" in raw:
        parts = [_wrap_single_line(part, max_line_chars) for part in raw.splitlines() if part.strip()]
        return "\n".join(parts)
    return _wrap_single_line(raw, max_line_chars)


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


WORDS_JSON_SCHEMA_VERSION = 1


def default_words_json_path(srt_path: Path) -> Path:
    """Return default sidecar `.words.json` path beside an `.srt` path (e.g. `foo.srt` -> `foo.words.json`)."""
    if srt_path.name.lower().endswith(".words.json"):
        return srt_path
    return srt_path.with_name(f"{srt_path.stem}.words.json")


def find_sidecar_words_json(srt_path: Path) -> Path | None:
    """Locate an existing sidecar `.words.json` beside an `.srt` file, or return None."""
    if srt_path.name.lower().endswith(".words.json"):
        return srt_path if srt_path.is_file() else None
    cand = default_words_json_path(srt_path)
    return cand if cand.is_file() else None


def save_words_json(
    words: list[dict[str, Any]],
    path: Path,
    *,
    provider: str = "",
    model: str = "",
    source_media: str = "",
) -> Path:
    """Persist original-timeline word-level timestamps to a compact `.words.json` sidecar file."""
    cleaned = clamp_stretched_words(words)
    serialized_words: list[dict[str, Any]] = []
    for wid, w in enumerate(cleaned, 1):
        txt = str(w.get("text") or "").strip()
        if not txt:
            continue
        item: dict[str, Any] = {
            "id": len(serialized_words) + 1,
            "seg": int(w.get("seg", 0)),
            "start": round(max(0.0, float(w.get("start", 0.0))), 3),
            "end": round(max(0.0, float(w.get("end", 0.0))), 3),
            "text": txt,
            "space_before": bool(w.get("space_before", True)),
        }
        if item["end"] <= item["start"]:
            item["end"] = round(item["start"] + 0.05, 3)
        spk = w.get("speaker") or w.get("speaker_id")
        if spk:
            item["speaker"] = str(spk)
        serialized_words.append(item)

    payload = {
        "schema_version": WORDS_JSON_SCHEMA_VERSION,
        "provider": str(provider or ""),
        "model": str(model or ""),
        "source_media": str(source_media or ""),
        "word_count": len(serialized_words),
        "words": serialized_words,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_words_json(path: Path) -> list[dict[str, Any]]:
    """Load and normalize word-level timestamps from a `.words.json` file with sequential 1-based `id`s."""
    if not path.is_file():
        raise FileNotFoundError(f"找不到词级时间戳文件: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw_words = raw.get("words")
    elif isinstance(raw, list):
        raw_words = raw
    else:
        raw_words = None
    if not isinstance(raw_words, list):
        raise ValueError(f"无效的 .words.json 格式（缺少 words 列表）: {path}")

    parsed: list[dict[str, Any]] = []
    for item in raw_words:
        if not isinstance(item, dict):
            continue
        txt = str(item.get("text") or "").strip()
        if not txt:
            continue
        ws = max(0.0, float(item.get("start", 0.0)))
        we = max(ws + 0.05, float(item.get("end", ws + 0.05)))
        entry: dict[str, Any] = {
            "seg": int(item.get("seg", 0)),
            "start": ws,
            "end": we,
            "text": txt,
            "space_before": bool(item.get("space_before", True)),
        }
        spk = item.get("speaker") or item.get("speaker_id")
        if spk:
            entry["speaker"] = str(spk)
        parsed.append(entry)

    cleaned = clamp_stretched_words(parsed)
    for wid, entry in enumerate(cleaned, 1):
        entry["id"] = wid
    return cleaned


def reconstruct_cues_from_word_spans(
    words: list[dict[str, Any]],
    spans: list[tuple[int, int]],
) -> list[SrtCue]:
    """Deterministically build monotonic non-overlapping `SrtCue`s from 1-based inclusive word ID spans."""
    if not words or not spans:
        return []
    by_id: dict[int, int] = {}
    for idx, w in enumerate(words):
        wid = int(w.get("id", idx + 1))
        by_id[wid] = idx

    cues: list[SrtCue] = []
    for start_wid, end_wid in spans:
        s_idx = by_id.get(int(start_wid))
        e_idx = by_id.get(int(end_wid))
        if s_idx is None or e_idx is None or e_idx < s_idx:
            continue
        buf = words[s_idx : e_idx + 1]
        if not buf:
            continue
        start = max(0.0, float(buf[0]["start"]))
        end = float(buf[-1]["end"])
        if cues and start < cues[-1].end:
            start = cues[-1].end
        if end <= start:
            end = start + 0.15
        text = _join_word_buf(buf).strip()
        if not text:
            continue
        cues.append(SrtCue(id=len(cues) + 1, start=round(start, 3), end=round(end, 3), text=text))
    return cues
