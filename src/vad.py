"""Local speech-activity detection (pure-ONNX Silero VAD + ffmpeg silencedetect fallback) and chunk packing."""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path

SR = 16000
SILERO_WINDOW = 512
SILERO_CONTEXT = 64


def find_silero_onnx_model(explicit_path: Path | None = None) -> Path | None:
    if explicit_path and explicit_path.is_file():
        return explicit_path
    env_p = os.environ.get("SILERO_VAD_ONNX", "").strip()
    if env_p and Path(env_p).is_file():
        return Path(env_p)
    spec = importlib.util.find_spec("silero_vad")
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            cand = Path(loc) / "data" / "silero_vad.onnx"
            if cand.is_file():
                return cand
    return None


def detect_speech_segments_silero(
    pcm_path: Path,
    *,
    onnx_path: Path,
    threshold: float = 0.35,
    neg_threshold: float = 0.20,
    min_speech_ms: int = 200,
    min_silence_ms: int = 500,
    speech_pad_ms: int = 500,
    merge_gap_ms: int = 600,
) -> tuple[list[tuple[float, float]], float]:
    """Streaming pure-ONNX Silero VAD (no PyTorch, < 80 MB peak RAM, suitable for 1C1G nodes)."""
    import numpy as np
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, SILERO_CONTEXT), dtype=np.float32)
    sr_arr = np.array(SR, dtype=np.int64)

    total_samples = pcm_path.stat().st_size // 4
    min_speech_samples = int(SR * min_speech_ms / 1000)
    min_silence_samples = int(SR * min_silence_ms / 1000)

    triggered = False
    temp_end = 0
    cur_start = 0
    raw_segments: list[list[int]] = []
    block_samples = SILERO_WINDOW * 1024  # ~32.7s per streaming block (~2 MB RAM)
    sample_cursor = 0

    with pcm_path.open("rb") as fh:
        while True:
            raw = fh.read(block_samples * 4)
            if not raw:
                break
            arr = np.frombuffer(raw, dtype=np.float32)
            n_win = len(arr) // SILERO_WINDOW
            for i in range(n_win):
                chunk = arr[i * SILERO_WINDOW : (i + 1) * SILERO_WINDOW].reshape(1, SILERO_WINDOW)
                inp = np.concatenate([context, chunk], axis=1)
                out, state = sess.run(None, {"input": inp, "state": state, "sr": sr_arr})
                context = chunk[:, -SILERO_CONTEXT:]
                prob = float(out[0][0])
                pos = sample_cursor + i * SILERO_WINDOW

                if prob >= threshold and temp_end:
                    temp_end = 0
                if prob >= threshold and not triggered:
                    triggered = True
                    cur_start = pos
                    continue
                if prob < neg_threshold and triggered:
                    if not temp_end:
                        temp_end = pos
                    if pos - temp_end >= min_silence_samples:
                        if temp_end - cur_start >= min_speech_samples:
                            raw_segments.append([cur_start, temp_end])
                        cur_start = 0
                        temp_end = 0
                        triggered = False
            sample_cursor += len(arr)

    if triggered and (total_samples - cur_start >= min_speech_samples):
        raw_segments.append([cur_start, total_samples])

    pad_samples = int(SR * speech_pad_ms / 1000)
    merge_gap_samples = int(SR * merge_gap_ms / 1000)
    merged: list[list[int]] = []
    for s, e in raw_segments:
        ps = max(0, s - pad_samples)
        pe = min(total_samples, e + pad_samples)
        if merged and ps <= merged[-1][1] + merge_gap_samples:
            merged[-1][1] = max(merged[-1][1], pe)
        else:
            merged.append([ps, pe])

    return [(round(s / SR, 3), round(e / SR, 3)) for s, e in merged], total_samples / SR


def detect_speech_segments_silencedetect(
    pcm_path: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    noise_db: float = -30.0,
    min_silence_s: float = 1.5,
    speech_pad_s: float = 0.5,
) -> tuple[list[tuple[float, float]], float]:
    """Zero-dependency ffmpeg silencedetect fallback when onnxruntime/silero_vad is unavailable."""
    total_dur = (pcm_path.stat().st_size // 4) / SR
    res = subprocess.run(
        [
            ffmpeg_bin, "-hide_banner", "-nostdin",
            "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(pcm_path),
            "-af", f"silencedetect=noise={noise_db:g}dB:d={min_silence_s:g}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, check=False,
    )
    silences: list[tuple[float, float]] = []
    cur_start: float | None = None
    for line in res.stderr.splitlines():
        m1 = re.search(r"silence_start:\s*([0-9.]+)", line)
        m2 = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m1:
            cur_start = float(m1.group(1))
        if m2 and cur_start is not None:
            silences.append((cur_start, float(m2.group(1))))
            cur_start = None
    if cur_start is not None:
        silences.append((cur_start, total_dur))

    kept: list[list[float]] = []
    pos = 0.0
    for s0, s1 in silences:
        a = max(0.0, pos - speech_pad_s)
        b = min(total_dur, s0 + speech_pad_s)
        if b > a:
            if kept and a <= kept[-1][1]:
                kept[-1][1] = max(kept[-1][1], b)
            else:
                kept.append([a, b])
        pos = s1
    if pos < total_dur:
        a = max(0.0, pos - speech_pad_s)
        b = total_dur
        if kept and a <= kept[-1][1]:
            kept[-1][1] = max(kept[-1][1], b)
        else:
            kept.append([a, b])
    return [(round(a, 3), round(b, 3)) for a, b in kept], total_dur


def pack_segments_into_chunks(
    segments: list[tuple[float, float]],
    *,
    max_chunk_seconds: float = 1680.0,
    silence_gap_s: float = 0.5,
) -> list[list[dict[str, float | int]]]:
    """Pack speech segments into <= max_chunk_seconds chunks with exact [compact -> orig] maps."""
    if max_chunk_seconds <= 10.0:
        raise ValueError("max_chunk_seconds 必须大于 10 秒")
    chunks: list[list[dict[str, float | int]]] = []
    cur_chunk: list[dict[str, float | int]] = []
    cursor = 0.0
    global_idx = 0

    for orig_s, orig_e in segments:
        pos = orig_s
        while pos < orig_e:
            sub_end = min(orig_e, pos + max_chunk_seconds)
            dur = sub_end - pos
            needed = dur if not cur_chunk else (silence_gap_s + dur)
            if cur_chunk and cursor + needed > max_chunk_seconds:
                chunks.append(cur_chunk)
                cur_chunk = []
                cursor = 0.0
            if cur_chunk:
                cursor += silence_gap_s
            cur_chunk.append({
                "idx": global_idx,
                "compact_start": round(cursor, 4),
                "compact_end": round(cursor + dur, 4),
                "orig_start": round(pos, 4),
                "orig_end": round(sub_end, 4),
            })
            cursor += dur
            global_idx += 1
            pos = sub_end

    if cur_chunk:
        chunks.append(cur_chunk)
    return chunks


def render_chunk_mp3(
    pcm_path: Path,
    chunk_map: list[dict[str, float | int]],
    out_mp3: Path,
    *,
    ffmpeg_bin: str = "ffmpeg",
    silence_gap_s: float = 0.5,
) -> None:
    """Stream selected PCM segments into a compact MP3 with constant low memory usage."""
    gap_bytes = b"\x00" * (int(round(SR * silence_gap_s)) * 4)
    total_bytes = pcm_path.stat().st_size
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
        "-c:a", "libmp3lame", "-b:a", "48k", str(out_mp3),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        with pcm_path.open("rb") as src_fh:
            for i, seg in enumerate(chunk_map):
                start_byte = max(0, min(total_bytes, int(round(float(seg["orig_start"]) * SR)) * 4))
                end_byte = max(start_byte, min(total_bytes, int(round(float(seg["orig_end"]) * SR)) * 4))
                src_fh.seek(start_byte)
                remaining = end_byte - start_byte
                while remaining > 0:
                    block = src_fh.read(min(65536, remaining))
                    if not block:
                        break
                    proc.stdin.write(block)
                    remaining -= len(block)
                if i + 1 < len(chunk_map) and gap_bytes:
                    proc.stdin.write(gap_bytes)
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg 编码压缩音频失败 ({rc}): {stderr[:300]}")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
