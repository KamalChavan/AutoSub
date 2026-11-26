#!/usr/bin/env python3
"""AutoSub CLI: General-purpose VAD speech transcription and profile-driven subtitle translation."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.config import (
    DEFAULT_CONFIG_PATH,
    get_app_root,
    get_bundle_root,
    is_frozen,
    load_engine_config,
    resolve_api_key,
    resolve_binary,
)
from src.profile import list_profiles
from src.transcribe import run_transcript_cli, transcribe_media
from src.translate import run_translate_cli, translate_srt
from src.vad import probe_silero_vad_runtime


def _configure_console_utf8() -> None:
    """Prevent UnicodeEncodeError on Windows GBK/cp936 consoles when printing Chinese logs."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _add_transcript_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--vad",
        choices=["silero_onnx", "silencedetect", "none"],
        default=None,
        help="覆盖本地人声裁剪模式（默认读取 profile / config/defaults.yaml）",
    )
    p.add_argument(
        "--model",
        default=None,
        help="覆盖语音转写模型（默认 gemini-3.5-transcribe）",
    )
    p.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=None,
        help="单次转写音频分段上限秒数（默认 1680s，需 <= 1800s 以支持词级时间戳）",
    )


def _add_translate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="向 Profile 的 Prompt 模板注入上下文变量（可重复，例如 --var hero=死灵龙 --var streamer=4oapm）",
    )
    p.add_argument(
        "--glossary",
        action="append",
        default=[],
        metavar="PATH",
        help="附加外部术语表 YAML 文件（可重复，与 Profile 自带术语表合并）",
    )
    p.add_argument(
        "--target-lang",
        default=None,
        help="覆盖目标翻译语言（如 'Simplified Chinese' / 'English' / 'Japanese'）",
    )
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--bilingual",
        dest="bilingual",
        action="store_true",
        default=None,
        help="强制输出双语字幕",
    )
    mode_group.add_argument(
        "--mono",
        dest="bilingual",
        action="store_false",
        help="仅输出目标语言单语字幕",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="覆盖每批翻译条数上限（默认 280，自动均匀切分）",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autosub",
        description="通用视频/音频本地 VAD 语音转录 + Profile 驱动双语字幕翻译工具",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="全局默认引擎配置路径 (默认: config/defaults.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="子命令")

    # 1. transcript
    tr_parser = subparsers.add_parser(
        "transcript",
        help="将视频/音频通过本地 VAD 裁剪并转写为原语 SRT 字幕",
    )
    tr_parser.add_argument("input", help="输入视频或音频文件路径（.mp4/.mkv/.ts/.mp3/.wav 等）")
    tr_parser.add_argument(
        "-p",
        "--profile",
        default=None,
        help="可选：Profile 名称或外部 YAML 路径（用于读取专属 VAD/分段参数）",
    )
    _add_transcript_args(tr_parser)
    tr_parser.add_argument("-o", "--output", default=None, help="输出原语 .srt 路径（默认同目录同名 .srt）")

    # 2. translate
    tl_parser = subparsers.add_parser(
        "translate",
        help="根据 Profile 与术语表对原语 SRT 执行去噪、ASR 纠偏与单/双语翻译",
    )
    tl_parser.add_argument("input", help="输入原语 .srt 文件路径")
    tl_parser.add_argument(
        "-p",
        "--profile",
        default=None,
        help="Profile 名称（如 default / twitch_dota2）或任意外部 YAML 文件路径",
    )
    _add_translate_args(tl_parser)
    tl_parser.add_argument("-o", "--output", default=None, help="输出翻译后 .srt 路径")

    # 3. run (one-shot transcript -> translate)
    run_parser = subparsers.add_parser(
        "run",
        help="一键串行执行 transcript（提取原语 SRT）-> translate（翻译为目标/双语 SRT）",
    )
    run_parser.add_argument("input", help="输入视频或音频文件路径")
    run_parser.add_argument(
        "-p",
        "--profile",
        default=None,
        help="Profile 名称（如 default / twitch_dota2）或任意外部 YAML 文件路径",
    )
    _add_transcript_args(run_parser)
    _add_translate_args(run_parser)
    run_parser.add_argument("--srt-output", default=None, help="可选：中间原语 .srt 保存路径")
    run_parser.add_argument("-o", "--output", default=None, help="可选：最终翻译 .srt 保存路径")

    # 4. profiles
    subparsers.add_parser(
        "profiles",
        help="列出所有内置 Profile 及其可配置的上下文变量",
    )

    # 5. doctor
    doc_parser = subparsers.add_parser(
        "doctor",
        help="检查本地运行环境（FFmpeg、Silero VAD ONNX、Profile 与 API Key 配置状态）",
    )
    doc_parser.add_argument(
        "--strict-bundle",
        action="store_true",
        help="打包验收模式：要求 FFmpeg、Silero ONNX 模型与内置 Profile 均可用（不要求配置 API Key）",
    )

    return parser


def run_doctor(defaults_path: Path, *, strict_bundle: bool = False) -> int:
    defaults = load_engine_config(defaults_path)
    tools_cfg = defaults.get("tools") or {}
    api_cfg = defaults.get("api") or {}

    ffmpeg_resolved = resolve_binary(str(tools_cfg.get("ffmpeg_bin") or ""), ["ffmpeg.exe", "ffmpeg"])
    ffmpeg_ok = Path(ffmpeg_resolved).is_file() or shutil.which(ffmpeg_resolved) is not None

    silero_ok, silero_backend, onnx_path, ort_lib = probe_silero_vad_runtime()

    profiles = list_profiles()
    profiles_ok = len(profiles) > 0
    api_key_ok = bool(resolve_api_key(str(api_cfg.get("api_key") or "")))

    print("=== [AutoSub 环境自检] ===", flush=True)
    print(f"运行模式: {'独立打包版 (Frozen)' if is_frozen() else 'Python 源码模式'}", flush=True)
    print(f"程序目录: {get_app_root()}", flush=True)
    if get_bundle_root() != get_app_root():
        print(f"内置资源: {get_bundle_root()}", flush=True)
    print(f"FFmpeg: {'OK' if ffmpeg_ok else 'MISSING'} ({ffmpeg_resolved})", flush=True)
    print(
        f"Silero VAD (ONNX): {'OK' if silero_ok else '降级为 silencedetect'} "
        f"(backend={silero_backend}, runtime={ort_lib or 'n/a'}, model={onnx_path or '未找到'})",
        flush=True,
    )
    print(f"内置 Profile: {', '.join(n for n, _, _ in profiles) or '未找到'}", flush=True)
    print(f"Gemini API Key: {'已配置' if api_key_ok else '未配置（请设置 GEMINI_API_KEY 或创建 .env）'}", flush=True)

    if strict_bundle and not (ffmpeg_ok and silero_ok and profiles_ok):
        print("[自检失败] 打包验收模式要求 FFmpeg、Silero ONNX 与内置 Profile 全部就绪。", flush=True)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    _configure_console_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    defaults_path = Path(args.config).resolve()

    if args.command == "doctor":
        return run_doctor(defaults_path, strict_bundle=args.strict_bundle)

    if args.command == "profiles":
        items = list_profiles()
        if not items:
            print("未找到内置 Profile。")
            return 0
        print("可用内置 Profile（也可直接传入任意外部 .yaml 路径）:")
        for name, desc, dvars in items:
            var_keys = ", ".join(dvars.keys()) if dvars else "(无)"
            print(f"  - {name}: {desc}")
            print(f"      默认 --var 变量: {var_keys}")
        return 0

    if args.command == "transcript":
        return run_transcript_cli(
            Path(args.input).resolve(),
            defaults_path=defaults_path,
            profile_name=args.profile,
            output_path=Path(args.output).resolve() if args.output else None,
            vad_override=args.vad,
            model_override=args.model,
            max_chunk_seconds_override=args.max_chunk_seconds,
        )

    if args.command == "translate":
        return run_translate_cli(
            Path(args.input).resolve(),
            defaults_path=defaults_path,
            profile_name=args.profile,
            vars_list=args.var,
            extra_glossaries=args.glossary,
            target_language_override=args.target_lang,
            bilingual_override=args.bilingual,
            batch_size_override=args.batch_size,
            output_path=Path(args.output).resolve() if args.output else None,
        )

    if args.command == "run":
        try:
            raw_srt = transcribe_media(
                Path(args.input).resolve(),
                defaults_path=defaults_path,
                profile_name=args.profile,
                output_path=Path(args.srt_output).resolve() if args.srt_output else None,
                vad_override=args.vad,
                model_override=args.model,
                max_chunk_seconds_override=args.max_chunk_seconds,
            )
            if raw_srt is None:
                return 0
            translate_srt(
                raw_srt,
                defaults_path=defaults_path,
                profile_name=args.profile,
                vars_list=args.var,
                extra_glossaries=args.glossary,
                target_language_override=args.target_lang,
                bilingual_override=args.bilingual,
                batch_size_override=args.batch_size,
                output_path=Path(args.output).resolve() if args.output else None,
            )
            return 0
        except Exception as exc:
            print(f"[错误] {exc}", flush=True)
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
