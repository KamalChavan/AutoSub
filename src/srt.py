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


def _join_word_buf(buf: list[dict[str, Any]]) -> str:
    if not buf:
        return ""
    parts = [str(buf[0]["text"])]
    for w in buf[1:]:
        sep = " " if w.get("space_before", True) else ""
        parts.append(sep + str(w["text"]))
    return "".join(parts)


def group_words_to_cues(words: list[dict[str, Any]]) -> list[SrtCue]:
    """Group word-level timestamps into readable SRT cues without crossing VAD segment or speaker boundaries."""
    if not words:
        return []
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

    for w in words:
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
                flush()
            elif len(cur_text) + next_sep_len + len(w["text"]) > 76 or (float(w["end"]) - float(buf[0]["start"])) > 6.0:
                flush()
        buf.append(w)
    flush()
    return cues
