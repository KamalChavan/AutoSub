"""Gemini speech-to-text transcription pipeline (`autosub transcript`).

Decodes media to 16kHz mono PCM, runs local VAD compaction (`silero_onnx` / `silencedetect` / `none`),
uploads compact chunks via Gemini Files API, transcribes via `/v1beta/interactions`
(`gemini-3.5-transcribe`, verbatim + word timestamps), remaps timestamps back to the original
timeline, and writes `.srt` incrementally after each chunk.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from src.config import DEFAULT_CONFIG_PATH, load_engine_config, resolve_api_key, resolve_binary
from src.http_client import GeminiApiError, delete_url, post_json, upload_resumable_file
from src.profile import load_profile
from src.srt import append_srt_cues, group_words_to_cues, map_word_to_original
from src.vad import (
    SR,
    detect_speech_segments_silencedetect,
    detect_speech_segments_silero,
    pack_segments_into_chunks,
    probe_silero_vad_runtime,
    render_chunk_mp3,
)

DEFAULT_TRANSCRIBE_MODEL = "gemini-3.5-transcribe"


def upload_audio_to_files_api(
    audio_path: Path,
    *,
    api_key: str,
    api_base: str = "https://generativelanguage.googleapis.com/v1beta",
    proxy: str = "",
) -> dict[str, Any]:
    return upload_resumable_file(
        audio_path,
        mime_type="audio/mp3",
        api_key=api_key,
        api_base=api_base,
        proxy=proxy,
        timeout=300,
    )


def delete_remote_file(
    file_name: str,
    *,
    api_key: str,
    api_base: str = "https://generativelanguage.googleapis.com/v1beta",
    proxy: str = "",
) -> None:
    if not file_name:
        return
    delete_url(
        f"{api_base.rstrip('/')}/{file_name}",
        api_key=api_key,
        proxy=proxy,
        timeout=30,
    )


def call_gemini_transcribe(
    file_uri: str,
    *,
    api_key: str,
    model: str = DEFAULT_TRANSCRIBE_MODEL,
    api_base: str = "https://generativelanguage.googleapis.com/v1beta",
    proxy: str = "",
    max_retries: int = 6,
    sleep_fn: Callable[[float], None] = time.sleep,
    post_fn: Callable[..., dict[str, Any]] | None = None,
    progress_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Call `/v1beta/interactions` with verbatim mode + word-level timestamps."""
    model_id = model.removeprefix("google/").removeprefix("models/")
    payload = {
        "model": model_id,
        "input": [{"type": "audio", "uri": file_uri, "mime_type": "audio/mp3"}],
        "generation_config": {
            "transcription_config": {
                "mode": {
                    "type": "verbatim",
                    "timestamp_granularities": ["word"],
                }
            }
        },
    }
    url = f"{api_base.rstrip('/')}/interactions"
    requester = post_fn or post_json
    last_err: Exception | None = None

    for attempt in range(1, max(1, max_retries) + 1):
        try:
            return requester(url, payload, api_key=api_key, proxy=proxy, timeout=360)
        except GeminiApiError as api_err:
            last_err = api_err
            if api_err.is_429_rate_limit and api_err.retry_after_seconds is not None and attempt < max_retries:
                wait_s = api_err.retry_after_seconds + 3
                if progress_fn:
                    progress_fn(f"    [转写限流 {attempt}/{max_retries}] 触发 429，等待 {wait_s}s 后重试...")
                sleep_fn(wait_s)
                continue
            if api_err.is_503_transient and attempt < max_retries:
                wait_s = min(30, 8 * attempt)
                if progress_fn:
                    progress_fn(f"    [转写高负载 {attempt}/{max_retries}] 遇到 {api_err.code}，{wait_s}s 后重试...")
                sleep_fn(wait_s)
                continue
            raise
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                wait_s = min(30, 8 * attempt)
                if progress_fn:
                    progress_fn(f"    [转写重试 {attempt}/{max_retries}] 网络/异常 ({type(exc).__name__})，{wait_s}s 后重试...")
                sleep_fn(wait_s)
                continue
            raise
    raise RuntimeError(f"gemini-3.5-transcribe 调用失败: {last_err}")


def extract_words_from_interaction(
    resp: dict[str, Any],
    seg_map: list[dict[str, float | int]],
) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for step in resp.get("steps") or []:
        for content in step.get("content") or []:
            for ann in content.get("annotations") or []:
                if ann.get("type") != "word_info":
                    continue
                txt = str(ann.get("text") or "").strip()
                if not txt:
                    continue
                ws = float(str(ann.get("start_offset", "0s")).rstrip("s"))
                we = float(str(ann.get("end_offset", "0s")).rstrip("s"))
                seg_idx, os_, oe_ = map_word_to_original(ws, we, seg_map)
                words.append({"seg": seg_idx, "start": os_, "end": oe_, "text": txt})
    return words


def transcribe_media(
    input_path: Path,
    *,
    defaults_path: Path = DEFAULT_CONFIG_PATH,
    profile_name: str | None = None,
    output_path: Path | None = None,
    vad_override: str | None = None,
    model_override: str | None = None,
    max_chunk_seconds_override: float | None = None,
) -> Path | None:
    """Transcribe a video/audio file to an original-language SRT file; returns output path or None if silent."""
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入媒体文件: {input_path}")

    defaults = load_engine_config(defaults_path)
    tools_cfg = defaults.get("tools") or {}
    api_cfg = defaults.get("api") or {}
    tr_cfg = dict(defaults.get("transcript") or {})

    if profile_name:
        profile = load_profile(profile_name)
        tr_cfg.update(profile.transcript)

    api_key = resolve_api_key(str(api_cfg.get("api_key") or ""))
    if not api_key:
        raise RuntimeError("未检测到 Gemini API Key（请设置 GEMINI_API_KEY、创建 .env 或填写 config/defaults.yaml）")

    ffmpeg_bin = resolve_binary(str(tools_cfg.get("ffmpeg_bin") or ""), ["ffmpeg", "ffmpeg.exe"])
    vad_mode = str(vad_override or tr_cfg.get("vad_mode") or "silero_onnx").strip().lower()
    model = str(model_override or tr_cfg.get("model") or DEFAULT_TRANSCRIBE_MODEL).strip()
    max_chunk_s = float(max_chunk_seconds_override or tr_cfg.get("max_chunk_seconds", 1680))
    silence_gap_s = float(tr_cfg.get("vad_silence_gap_s", 0.50))
    api_base = str(api_cfg.get("api_base") or "https://generativelanguage.googleapis.com/v1beta").strip()
    proxy = str(api_cfg.get("proxy") or "").strip()
    max_retries = int(tr_cfg.get("max_retries", 6))

    with tempfile.TemporaryDirectory(prefix="autosub_transcript_") as tmp:
        tmp_dir = Path(tmp)
        pcm_path = tmp_dir / "audio_16k.f32"
        print(f"[1/4] 正在抽取 16kHz 单声道音频: {input_path.name} ...", flush=True)
        subprocess.run(
            [
                ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", str(input_path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", str(pcm_path),
            ],
            check=True,
        )
        total_dur = (pcm_path.stat().st_size // 4) / SR

        effective_vad = vad_mode
        silero_ok, silero_backend, onnx_path, ort_lib = probe_silero_vad_runtime() if effective_vad == "silero_onnx" else (False, "", None, None)
        if effective_vad == "silero_onnx" and not silero_ok:
            print("[提示] 未检测到 ONNX Runtime 或 silero_vad.onnx，自动降级为 ffmpeg silencedetect", flush=True)
            effective_vad = "silencedetect"

        if effective_vad == "silero_onnx":
            assert onnx_path is not None
            print(f"[2/4] 正在运行本地 Silero VAD (ONNX / {silero_backend}) 裁剪无人声片段...", flush=True)
            segments, total_dur = detect_speech_segments_silero(
                pcm_path,
                onnx_path=onnx_path,
                ort_lib_path=ort_lib,
                threshold=float(tr_cfg.get("vad_threshold", 0.35)),
                neg_threshold=float(tr_cfg.get("vad_neg_threshold", 0.20)),
                speech_pad_ms=int(tr_cfg.get("vad_speech_pad_ms", 500)),
            )
        elif effective_vad == "silencedetect":
            print("[2/4] 正在运行 ffmpeg silencedetect 裁剪静音片段...", flush=True)
            segments, total_dur = detect_speech_segments_silencedetect(
                pcm_path,
                ffmpeg_bin=ffmpeg_bin,
                noise_db=float(tr_cfg.get("silence_noise_db", -30)),
                min_silence_s=float(tr_cfg.get("silence_min_duration_s", 1.5)),
                speech_pad_s=int(tr_cfg.get("vad_speech_pad_ms", 500)) / 1000.0,
            )
        else:
            print("[2/4] 已跳过 VAD（全量音频模式）", flush=True)
            segments = [(0.0, total_dur)]

        if not segments:
            print("[警告] 未检测到有效语音片段，跳过转写。", flush=True)
            return None

        chunks = pack_segments_into_chunks(segments, max_chunk_seconds=max_chunk_s, silence_gap_s=silence_gap_s)
        compact_total = sum(float(ch[-1]["compact_end"]) for ch in chunks)
        print(
            f"[3/4] 原始时长 {total_dur/60:.1f} 分钟 -> 压缩后 {compact_total/60:.1f} 分钟 "
            f"(节省 {max(0.0, 1 - compact_total/max(1.0, total_dur)):.1%})，共分 {len(chunks)} 段调用 {model}...",
            flush=True,
        )

        out_srt = output_path or input_path.with_suffix(".srt")
        out_srt.parent.mkdir(parents=True, exist_ok=True)
        out_srt.write_text("", encoding="utf-8")
        total_words = 0
        written_cues = 0
        for idx, ch_map in enumerate(chunks, 1):
            mp3_path = tmp_dir / f"chunk_{idx:02d}.mp3"
            print(f"  -> [分段 {idx}/{len(chunks)}] 正在编码并上传音频至 Gemini Files API...", flush=True)
            render_chunk_mp3(pcm_path, ch_map, mp3_path, ffmpeg_bin=ffmpeg_bin, silence_gap_s=silence_gap_s)
            finfo = upload_audio_to_files_api(mp3_path, api_key=api_key, api_base=api_base, proxy=proxy)
            try:
                print(f"  -> [分段 {idx}/{len(chunks)}] 正在调用 {model} 转写...", flush=True)
                t0 = time.perf_counter()
                resp = call_gemini_transcribe(
                    str(finfo["uri"]),
                    api_key=api_key,
                    model=model,
                    api_base=api_base,
                    proxy=proxy,
                    max_retries=max_retries,
                    progress_fn=lambda msg: print(msg, flush=True),
                )
                elapsed = round(time.perf_counter() - t0, 2)
                words = extract_words_from_interaction(resp, ch_map)
                total_words += len(words)
                chunk_cues = group_words_to_cues(words)
                if chunk_cues:
                    written_cues = append_srt_cues(
                        [(c.start, c.end, [c.text]) for c in chunk_cues],
                        out_srt,
                        start_index=written_cues + 1,
                        truncate=(written_cues == 0),
                    )
                usage = resp.get("usage") or {}
                print(
                    f"  -> [分段 {idx}/{len(chunks)} 完成] 耗时 {elapsed}s | 词数 {len(words)} | "
                    f"已实时写入 {out_srt.name} (累计 {written_cues} 条) | "
                    f"Token: {usage.get('total_input_tokens')} in / {usage.get('total_output_tokens')} out",
                    flush=True,
                )
            finally:
                delete_remote_file(str(finfo.get("name") or ""), api_key=api_key, api_base=api_base, proxy=proxy)

        print(f"[4/4] 原语字幕已生成: {out_srt}（共 {total_words} 词 / {written_cues} 条 SRT）", flush=True)
        return out_srt


def run_transcript_cli(
    input_path: Path,
    *,
    defaults_path: Path = DEFAULT_CONFIG_PATH,
    profile_name: str | None = None,
    output_path: Path | None = None,
    vad_override: str | None = None,
    model_override: str | None = None,
    max_chunk_seconds_override: float | None = None,
) -> int:
    try:
        transcribe_media(
            input_path,
            defaults_path=defaults_path,
            profile_name=profile_name,
            output_path=output_path,
            vad_override=vad_override,
            model_override=model_override,
            max_chunk_seconds_override=max_chunk_seconds_override,
        )
        return 0
    except Exception as exc:
        print(f"[错误] {exc}", flush=True)
        return 1
