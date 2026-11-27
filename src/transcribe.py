"""Multi-provider speech-to-text transcription pipeline (`autosub transcript`).

Decodes media to 16kHz mono PCM, runs local VAD compaction (`silero_onnx` / `silencedetect` / `none`),
dispatches compact chunks to the configured `TranscribeProvider` (Gemini `gemini-3.5-transcribe`,
ElevenLabs `scribe_v3`, or timestamped OpenAI-compatible Whisper endpoints), remaps native word-level
timestamps back to the original timeline, and writes `.srt` incrementally after each chunk.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from src.config import (
    DEFAULT_CONFIG_PATH,
    get_providers_config,
    load_engine_config,
    resolve_binary,
)
from src.http_client import delete_url, upload_resumable_file
from src.profile import load_profile
from src.providers import (
    ModelSpec,
    TranscribedWord,
    build_provider_settings,
    create_transcribe_provider,
    parse_model_spec,
)
from src.providers.elevenlabs import DEFAULT_ELEVENLABS_TRANSCRIBE_MODEL
from src.providers.gemini import (
    DEFAULT_GEMINI_TRANSCRIBE_MODEL,
    call_gemini_interactions_transcribe,
    extract_gemini_words,
)
from src.srt import append_srt_cues, group_words_to_cues, map_word_to_original
from src.vad import (
    SR,
    detect_speech_segments_silencedetect,
    detect_speech_segments_silero,
    pack_segments_into_chunks,
    probe_silero_vad_runtime,
    render_chunk_mp3,
)

DEFAULT_TRANSCRIBE_PROVIDER = "elevenlabs"
DEFAULT_TRANSCRIBE_MODEL = DEFAULT_ELEVENLABS_TRANSCRIBE_MODEL
_MP3_48K_BYTES_PER_SEC_ESTIMATE = 6400.0  # 48 kbps = 6000 B/s + container overhead


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
    model: str = DEFAULT_GEMINI_TRANSCRIBE_MODEL,
    api_base: str = "https://generativelanguage.googleapis.com/v1beta",
    proxy: str = "",
    max_retries: int = 6,
    sleep_fn: Callable[[float], None] = time.sleep,
    post_fn: Callable[..., dict[str, Any]] | None = None,
    progress_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Call Gemini `/v1beta/interactions` with verbatim mode + word-level timestamps."""
    return call_gemini_interactions_transcribe(
        file_uri,
        api_key=api_key,
        model=model,
        api_base=api_base,
        proxy=proxy,
        max_retries=max_retries,
        sleep_fn=sleep_fn,
        post_fn=post_fn,
        progress_fn=progress_fn,
    )


def map_transcribed_words_to_original(
    words: Sequence[TranscribedWord],
    seg_map: list[dict[str, float | int]],
) -> list[dict[str, Any]]:
    """Map provider-returned chunk-local `TranscribedWord` objects back to original media timeline."""
    mapped: list[dict[str, Any]] = []
    for w in words:
        seg_idx, os_, oe_ = map_word_to_original(w.start, w.end, seg_map)
        entry: dict[str, Any] = {
            "seg": seg_idx,
            "start": os_,
            "end": oe_,
            "text": w.text,
            "space_before": w.space_before,
        }
        if w.speaker_id:
            entry["speaker"] = w.speaker_id
        mapped.append(entry)
    return mapped


def extract_words_from_interaction(
    resp: dict[str, Any],
    seg_map: list[dict[str, float | int]],
) -> list[dict[str, Any]]:
    """Extract and remap words from a Gemini `/v1beta/interactions` payload."""
    words = extract_gemini_words(resp)
    return map_transcribed_words_to_original(words, seg_map)


def _default_model_for_provider(provider_name: str, protocol: str) -> str:
    if protocol == "elevenlabs":
        return DEFAULT_ELEVENLABS_TRANSCRIBE_MODEL
    if protocol == "gemini":
        return DEFAULT_GEMINI_TRANSCRIBE_MODEL
    if protocol == "openai":
        return "whisper-1"
    return DEFAULT_TRANSCRIBE_MODEL


def resolve_dual_asr_spec(
    tr_cfg: dict[str, Any],
    providers_cfg: dict[str, dict[str, Any]],
    *,
    primary_spec: ModelSpec,
    ref_enabled_override: bool | None = None,
    ref_provider_override: str | None = None,
    ref_model_override: str | None = None,
) -> ModelSpec | None:
    """Resolve optional reference ASR `ModelSpec` from YAML (`transcript.ref_*`) and CLI overrides."""
    if ref_enabled_override is False:
        return None
    enabled = (
        True
        if (ref_enabled_override is True or ref_model_override or ref_provider_override)
        else bool(tr_cfg.get("ref_enabled", False))
    )
    if not enabled:
        return None

    cfg_ref_provider = str(tr_cfg.get("ref_provider") or "").strip().lower()
    raw_ref_model = str(ref_model_override or tr_cfg.get("ref_model") or "").strip()
    eff_ref_provider = (ref_provider_override or cfg_ref_provider).strip().lower()

    if not raw_ref_model and eff_ref_provider:
        proto = str((providers_cfg.get(eff_ref_provider) or {}).get("protocol") or eff_ref_provider).strip().lower()
        raw_ref_model = _default_model_for_provider(eff_ref_provider, proto)

    if not raw_ref_model:
        raise ValueError(
            "已启用双 ASR (transcript.ref_enabled=true)，但未配置参考模型；"
            "请在 config/defaults.yaml 中填写 `transcript.ref_model`（例如 `gemini-3.5-transcribe`）"
            "或通过 `--ref-model` 指定"
        )

    ref_spec = parse_model_spec(
        raw_ref_model,
        default_provider=eff_ref_provider or "gemini",
        known_providers=providers_cfg,
        provider_override=ref_provider_override or (cfg_ref_provider or None),
    )
    if ref_spec.provider == primary_spec.provider and ref_spec.model == primary_spec.model:
        print(
            f"[提示] 双 ASR 参考轨 ({ref_spec.provider}/{ref_spec.model}) 与主转录模型相同，本轮自动跳过重复转录。",
            flush=True,
        )
        return None
    return ref_spec


def transcribe_media_pair(
    input_path: Path,
    *,
    defaults_path: Path = DEFAULT_CONFIG_PATH,
    profile_name: str | None = None,
    output_path: Path | None = None,
    ref_output_path: Path | None = None,
    vad_override: str | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    ref_enabled_override: bool | None = None,
    ref_provider_override: str | None = None,
    ref_model_override: str | None = None,
    max_chunk_seconds_override: float | None = None,
    language_override: str | None = None,
    secondary_languages_override: list[str] | None = None,
    keyterms_override: list[str] | None = None,
    diarize_override: bool | None = None,
    no_verbatim_override: bool | None = None,
    numbers_format_override: str | None = None,
) -> tuple[Path | None, Path | None]:
    """Transcribe media to primary SRT (and optional reference ASR SRT when dual-ASR is enabled).

    Runs FFmpeg decode, Silero VAD, and MP3 chunk encoding ONCE, then dispatches primary + reference
    providers concurrently per chunk. Returns `(primary_srt_path, ref_srt_path_or_None)`.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入媒体文件: {input_path}")

    defaults = load_engine_config(defaults_path)
    tools_cfg = defaults.get("tools") or {}
    providers_cfg = get_providers_config(defaults)
    tr_cfg = dict(defaults.get("transcript") or {})

    if profile_name:
        profile = load_profile(profile_name)
        tr_cfg.update(profile.transcript)

    cfg_provider = str(tr_cfg.get("provider") or DEFAULT_TRANSCRIBE_PROVIDER).strip().lower()
    raw_model = str(model_override or tr_cfg.get("model") or "").strip()
    if provider_override and not model_override:
        ov_key = provider_override.strip().lower()
        ov_proto = str((providers_cfg.get(ov_key) or {}).get("protocol") or ov_key).strip().lower()
        if ov_key != cfg_provider:
            raw_model = _default_model_for_provider(ov_key, ov_proto)
    if not raw_model:
        eff_key = (provider_override or cfg_provider).strip().lower()
        eff_proto = str((providers_cfg.get(eff_key) or {}).get("protocol") or eff_key).strip().lower()
        raw_model = _default_model_for_provider(eff_key, eff_proto)

    spec = parse_model_spec(
        raw_model,
        default_provider=cfg_provider,
        known_providers=providers_cfg,
        provider_override=provider_override,
    )
    settings = build_provider_settings(spec.provider, defaults, require_api_key=True)
    provider = create_transcribe_provider(settings)

    ref_spec = resolve_dual_asr_spec(
        tr_cfg,
        providers_cfg,
        primary_spec=spec,
        ref_enabled_override=ref_enabled_override,
        ref_provider_override=ref_provider_override,
        ref_model_override=ref_model_override,
    )
    ref_provider = None
    if ref_spec is not None:
        ref_settings = build_provider_settings(ref_spec.provider, defaults, require_api_key=True)
        ref_provider = create_transcribe_provider(ref_settings)

    ffmpeg_bin = resolve_binary(str(tools_cfg.get("ffmpeg_bin") or ""), ["ffmpeg", "ffmpeg.exe"])
    vad_mode = str(vad_override or tr_cfg.get("vad_mode") or "silero_onnx").strip().lower()
    max_chunk_s = float(max_chunk_seconds_override or tr_cfg.get("max_chunk_seconds", 1680))
    for prov in (provider, ref_provider):
        if prov is not None and prov.max_chunk_bytes is not None:
            max_by_size = max(30.0, prov.max_chunk_bytes / _MP3_48K_BYTES_PER_SEC_ESTIMATE)
            max_chunk_s = min(max_chunk_s, max_by_size)
    silence_gap_s = float(tr_cfg.get("vad_silence_gap_s", 0.50))
    max_retries = int(tr_cfg.get("max_retries", 6))

    # Build provider-specific transcription options (defaults -> profile -> CLI overrides)
    cfg_secondary = tr_cfg.get("secondary_language_codes") or []
    if isinstance(cfg_secondary, str):
        cfg_secondary = [c.strip() for c in cfg_secondary.split(",") if c.strip()]
    effective_secondary = list(secondary_languages_override) if secondary_languages_override else list(cfg_secondary)

    cfg_keyterms = tr_cfg.get("keyterms") or []
    if isinstance(cfg_keyterms, str):
        cfg_keyterms = [k.strip() for k in cfg_keyterms.split(",") if k.strip()]
    effective_keyterms = list(cfg_keyterms) + list(keyterms_override or [])

    stt_options: dict[str, Any] = {
        "language_code": str(language_override if language_override is not None else tr_cfg.get("language_code") or "").strip(),
        "secondary_language_codes": effective_secondary,
        "keyterms": effective_keyterms,
        "tag_audio_events": bool(tr_cfg.get("tag_audio_events", False)),
        "diarize": bool(tr_cfg.get("diarize", False)) if diarize_override is None else bool(diarize_override),
        "num_speakers": tr_cfg.get("num_speakers"),
        "no_verbatim": bool(tr_cfg.get("no_verbatim", False)) if no_verbatim_override is None else bool(no_verbatim_override),
        "numbers_format": str(
            numbers_format_override if numbers_format_override is not None else tr_cfg.get("numbers_format") or ""
        ).strip(),
        "prompt": str(tr_cfg.get("prompt") or "").strip(),
    }

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
        silero_ok, silero_backend, onnx_path, ort_lib = (
            probe_silero_vad_runtime() if effective_vad == "silero_onnx" else (False, "", None, None)
        )
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
            return None, None

        chunks = pack_segments_into_chunks(segments, max_chunk_seconds=max_chunk_s, silence_gap_s=silence_gap_s)
        compact_total = sum(float(ch[-1]["compact_end"]) for ch in chunks)
        model_label = f"{spec.provider}/{spec.model}"
        dual_desc = (
            f" + 参考轨 {ref_spec.provider}/{ref_spec.model}（双 ASR 并行）"
            if ref_spec is not None and ref_provider is not None
            else ""
        )
        print(
            f"[3/4] 原始时长 {total_dur/60:.1f} 分钟 -> 压缩后 {compact_total/60:.1f} 分钟 "
            f"(节省 {max(0.0, 1 - compact_total/max(1.0, total_dur)):.1%})，共分 {len(chunks)} 段调用 {model_label}{dual_desc}...",
            flush=True,
        )

        out_srt = output_path or input_path.with_suffix(".srt")
        out_srt.parent.mkdir(parents=True, exist_ok=True)
        out_srt.write_text("", encoding="utf-8")

        ref_out_srt: Path | None = None
        if ref_spec is not None and ref_provider is not None:
            ref_out_srt = ref_output_path or out_srt.with_name(f"{out_srt.stem}_ref.srt")
            ref_out_srt.parent.mkdir(parents=True, exist_ok=True)
            ref_out_srt.write_text("", encoding="utf-8")

        total_words = 0
        written_cues = 0
        ref_total_words = 0
        ref_written_cues = 0
        ref_active = ref_provider is not None and ref_spec is not None and ref_out_srt is not None

        for idx, ch_map in enumerate(chunks, 1):
            mp3_path = tmp_dir / f"chunk_{idx:02d}.mp3"
            target_desc = (
                f"{spec.provider} ({spec.model}) + {ref_spec.provider} ({ref_spec.model})"
                if ref_active and ref_spec is not None
                else f"{spec.provider} ({spec.model})"
            )
            print(f"  -> [分段 {idx}/{len(chunks)}] 正在编码并上传音频至 {target_desc}...", flush=True)
            render_chunk_mp3(pcm_path, ch_map, mp3_path, ffmpeg_bin=ffmpeg_bin, silence_gap_s=silence_gap_s)
            t0 = time.perf_counter()

            ref_res = None
            if ref_active and ref_provider is not None and ref_spec is not None:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    fut_primary = pool.submit(
                        provider.transcribe_chunk,
                        mp3_path,
                        model=spec.model,
                        options=stt_options,
                        max_retries=max_retries,
                        progress_fn=lambda msg: print(msg, flush=True),
                    )
                    fut_ref = pool.submit(
                        ref_provider.transcribe_chunk,
                        mp3_path,
                        model=ref_spec.model,
                        options=stt_options,
                        max_retries=max_retries,
                        progress_fn=lambda msg: print(msg, flush=True),
                    )
                    res = fut_primary.result()
                    try:
                        ref_res = fut_ref.result()
                    except Exception as ref_exc:
                        print(
                            f"  -> [警告] 参考 ASR ({ref_spec.provider}/{ref_spec.model}) 分段 {idx} 失败，"
                            f"本轮后续分段降级为单轨: {ref_exc}",
                            flush=True,
                        )
                        ref_active = False
            else:
                res = provider.transcribe_chunk(
                    mp3_path,
                    model=spec.model,
                    options=stt_options,
                    max_retries=max_retries,
                    progress_fn=lambda msg: print(msg, flush=True),
                )

            elapsed = round(time.perf_counter() - t0, 2)
            words = map_transcribed_words_to_original(res.words, ch_map)
            total_words += len(words)
            chunk_cues = group_words_to_cues(words)
            if chunk_cues:
                written_cues = append_srt_cues(
                    [(c.start, c.end, [c.text]) for c in chunk_cues],
                    out_srt,
                    start_index=written_cues + 1,
                    truncate=(written_cues == 0),
                )
            usage_part = f" | {res.usage_summary}" if res.usage_summary else ""
            print(
                f"  -> [分段 {idx}/{len(chunks)} 主轨完成] 耗时 {elapsed}s | 词数 {len(words)} | "
                f"已实时写入 {out_srt.name} (累计 {written_cues} 条){usage_part}",
                flush=True,
            )

            if ref_res is not None and ref_out_srt is not None:
                ref_words = map_transcribed_words_to_original(ref_res.words, ch_map)
                ref_total_words += len(ref_words)
                ref_chunk_cues = group_words_to_cues(ref_words)
                if ref_chunk_cues:
                    ref_written_cues = append_srt_cues(
                        [(c.start, c.end, [c.text]) for c in ref_chunk_cues],
                        ref_out_srt,
                        start_index=ref_written_cues + 1,
                        truncate=(ref_written_cues == 0),
                    )
                ref_usage = f" | {ref_res.usage_summary}" if ref_res.usage_summary else ""
                print(
                    f"  -> [分段 {idx}/{len(chunks)} 参考轨完成] 词数 {len(ref_words)} | "
                    f"已实时写入 {ref_out_srt.name} (累计 {ref_written_cues} 条){ref_usage}",
                    flush=True,
                )

        print(f"[4/4] 原语字幕已生成: {out_srt}（共 {total_words} 词 / {written_cues} 条 SRT）", flush=True)
        final_ref_srt: Path | None = None
        if ref_out_srt is not None:
            if ref_written_cues > 0:
                final_ref_srt = ref_out_srt
                print(
                    f"[4/4] 双 ASR 参考轨已生成: {ref_out_srt}（共 {ref_total_words} 词 / {ref_written_cues} 条 SRT）",
                    flush=True,
                )
            else:
                try:
                    ref_out_srt.unlink(missing_ok=True)
                except OSError:
                    pass
        return out_srt, final_ref_srt


def transcribe_media(
    input_path: Path,
    *,
    defaults_path: Path = DEFAULT_CONFIG_PATH,
    profile_name: str | None = None,
    output_path: Path | None = None,
    vad_override: str | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    max_chunk_seconds_override: float | None = None,
    language_override: str | None = None,
    secondary_languages_override: list[str] | None = None,
    keyterms_override: list[str] | None = None,
    diarize_override: bool | None = None,
    no_verbatim_override: bool | None = None,
    numbers_format_override: str | None = None,
) -> Path | None:
    """Transcribe a video/audio file to an original-language SRT file; returns output path or None if silent."""
    primary_srt, _ = transcribe_media_pair(
        input_path,
        defaults_path=defaults_path,
        profile_name=profile_name,
        output_path=output_path,
        vad_override=vad_override,
        provider_override=provider_override,
        model_override=model_override,
        max_chunk_seconds_override=max_chunk_seconds_override,
        language_override=language_override,
        secondary_languages_override=secondary_languages_override,
        keyterms_override=keyterms_override,
        diarize_override=diarize_override,
        no_verbatim_override=no_verbatim_override,
        numbers_format_override=numbers_format_override,
    )
    return primary_srt


def run_transcript_cli(
    input_path: Path,
    *,
    defaults_path: Path = DEFAULT_CONFIG_PATH,
    profile_name: str | None = None,
    output_path: Path | None = None,
    ref_output_path: Path | None = None,
    vad_override: str | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    ref_enabled_override: bool | None = None,
    ref_provider_override: str | None = None,
    ref_model_override: str | None = None,
    max_chunk_seconds_override: float | None = None,
    language_override: str | None = None,
    secondary_languages_override: list[str] | None = None,
    keyterms_override: list[str] | None = None,
    diarize_override: bool | None = None,
    no_verbatim_override: bool | None = None,
    numbers_format_override: str | None = None,
) -> int:
    try:
        transcribe_media_pair(
            input_path,
            defaults_path=defaults_path,
            profile_name=profile_name,
            output_path=output_path,
            ref_output_path=ref_output_path,
            vad_override=vad_override,
            provider_override=provider_override,
            model_override=model_override,
            ref_enabled_override=ref_enabled_override,
            ref_provider_override=ref_provider_override,
            ref_model_override=ref_model_override,
            max_chunk_seconds_override=max_chunk_seconds_override,
            language_override=language_override,
            secondary_languages_override=secondary_languages_override,
            keyterms_override=keyterms_override,
            diarize_override=diarize_override,
            no_verbatim_override=no_verbatim_override,
            numbers_format_override=numbers_format_override,
        )
        return 0
    except Exception as exc:
        print(f"[错误] {exc}", flush=True)
        return 1
