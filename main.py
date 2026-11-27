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
    get_providers_config,
    is_frozen,
    load_engine_config,
    resolve_binary,
    resolve_provider_api_key,
)
from src.profile import list_profiles
from src.providers import provider_capabilities
from src.transcribe import run_transcript_cli, transcribe_media_pair
from src.translate import normalize_translate_model_specs, run_translate_cli, translate_srt
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


def _add_transcript_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--vad",
        choices=["silero_onnx", "silencedetect", "none"],
        default=None,
        help="覆盖本地人声裁剪模式（默认读取 profile / config/defaults.yaml）",
    )
    p.add_argument(
        "--max-chunk-seconds",
        type=float,
        default=None,
        help="单次转写音频分段上限秒数（默认 1680s；Gemini 词级时间戳上限 1800s，Whisper 类受 24MB 文件体积限制）",
    )
    p.add_argument(
        "--lang",
        default=None,
        help="可选：主语音语言代码（如 rus / eng / zho / jpn，ElevenLabs Scribe 与 Whisper 生效）",
    )
    p.add_argument(
        "--secondary-lang",
        action="append",
        default=[],
        metavar="CODE",
        help="可选：次语言代码（可重复，仅 ElevenLabs scribe_v3 混语场景生效，如 --secondary-lang eng）",
    )
    p.add_argument(
        "--keyterm",
        action="append",
        default=[],
        metavar="TERM",
        help="可选：专有名词热词（可重复，ElevenLabs Scribe 生效，注意会增加 +25%% 计费）",
    )
    p.add_argument(
        "--diarize",
        action="store_true",
        default=None,
        help="可选：开启说话人分离（ElevenLabs Scribe 生效，自动按说话人切换断句）",
    )
    p.add_argument(
        "--no-verbatim",
        action="store_true",
        default=None,
        help="可选：过滤口癖/填充词/结巴（ElevenLabs Scribe 生效）",
    )
    p.add_argument(
        "--numbers-format",
        choices=["digits", "words"],
        default=None,
        help="可选：数字输出格式（仅 ElevenLabs scribe_v3 生效：digits 或 words）",
    )
    p.add_argument(
        "--ref-model",
        "--ref-transcript-model",
        dest="ref_model",
        default=None,
        help="可选：临时启用双 ASR 并指定参考转录模型（如 gemini-3.5-transcribe；也可在 config/defaults.yaml 中配置 ref_enabled: true）",
    )
    p.add_argument(
        "--ref-provider",
        "--ref-transcript-provider",
        dest="ref_provider",
        default=None,
        help="可选：覆盖双 ASR 参考转录厂商（如 gemini / elevenlabs）",
    )
    p.add_argument(
        "--no-ref-transcript",
        dest="ref_enabled",
        action="store_false",
        default=None,
        help="可选：临时关闭 YAML 中启用的双 ASR 参考轨",
    )


def _add_translate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=None,
        help="覆盖 OpenAI GPT-6 推理深度（默认 low；gpt-6.2-sol 不支持 none）",
    )
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
    p.add_argument(
        "--auto-context",
        action="store_true",
        default=None,
        help="开启 Pass 0 全片上下文自动分析（提前提取英雄/人名/高频 ASR 错词并缓存复用于所有批次）",
    )
    p.add_argument(
        "--no-merge-cues",
        dest="merge_cues",
        action="store_false",
        default=None,
        help="关闭相邻短间隔碎句语义合并（默认开启）",
    )
    p.add_argument(
        "--no-polish-timing",
        dest="polish_timing",
        action="store_false",
        default=None,
        help="关闭翻译后 CPS 读速/最短驻留时间轴打磨（默认开启）",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autosub",
        description="通用视频/音频本地 VAD 多厂商语音转录 + Profile 驱动双语字幕翻译工具",
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
        help="可选：Profile 名称或外部 YAML 路径（用于读取专属 VAD/分段/语种参数）",
    )
    tr_parser.add_argument(
        "--provider",
        default=None,
        help="覆盖语音转录厂商（如 gemini / elevenlabs / openai / openai_compatible）",
    )
    tr_parser.add_argument(
        "--model",
        default=None,
        help="覆盖语音转写模型（支持 provider/model 前缀，如 gemini-3.5-transcribe / elevenlabs/scribe_v3）",
    )
    _add_transcript_common_args(tr_parser)
    tr_parser.add_argument("-o", "--output", default=None, help="输出原语 .srt 路径（默认同目录同名 .srt）")
    tr_parser.add_argument("--ref-output", default=None, help="可选：双 ASR 参考轨原语 .srt 保存路径（默认 <output>_ref.srt）")

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
    tl_parser.add_argument(
        "--provider",
        default=None,
        help="覆盖字幕翻译厂商（如 gemini / openai / openai_compatible 或自定义厂商名）",
    )
    tl_parser.add_argument(
        "--model",
        default=None,
        help="覆盖翻译模型或逗号分隔降级链（如 openai/gpt-6.2-sol,gemini/gemini-4-hydrogen）",
    )
    tl_parser.add_argument(
        "--ref-srt",
        default=None,
        metavar="PATH",
        help="可选：第二份独立 ASR 原语 .srt 参考轨（用于双 ASR 交叉纠偏，例如 ElevenLabs 主轨 + Gemini 参考轨）",
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
    run_parser.add_argument(
        "--transcript-provider",
        default=None,
        help="覆盖语音转录厂商（如 elevenlabs / gemini / openai）",
    )
    run_parser.add_argument(
        "--transcript-model",
        "--model",
        dest="transcript_model",
        default=None,
        help="覆盖语音转写模型（如 elevenlabs/scribe_v3 / gemini-3.5-transcribe；--model 为兼容别名）",
    )
    _add_transcript_common_args(run_parser)
    run_parser.add_argument(
        "--translate-provider",
        default=None,
        help="覆盖字幕翻译厂商（如 openai / gemini / openai_compatible）",
    )
    run_parser.add_argument(
        "--translate-model",
        default=None,
        help="覆盖翻译模型或逗号分隔降级链（如 openai/gpt-6.2-sol,gemini/gemini-4-hydrogen）",
    )
    _add_translate_args(run_parser)
    run_parser.add_argument("--srt-output", default=None, help="可选：中间原语 .srt 保存路径")
    run_parser.add_argument("--ref-srt-output", default=None, help="可选：双 ASR 参考轨中间 .srt 保存路径")
    run_parser.add_argument("-o", "--output", default=None, help="可选：最终翻译 .srt 保存路径")

    # 4. providers
    subparsers.add_parser(
        "providers",
        help="列出已配置的多厂商 API 协议、支持能力与 API Key 状态",
    )

    # 5. profiles
    subparsers.add_parser(
        "profiles",
        help="列出所有内置 Profile 及其可配置的上下文变量",
    )

    # 6. doctor
    doc_parser = subparsers.add_parser(
        "doctor",
        help="检查本地运行环境（FFmpeg、Silero VAD ONNX、Profile 与多厂商 API Key 配置状态）",
    )
    doc_parser.add_argument(
        "--strict-bundle",
        action="store_true",
        help="打包验收模式：要求 FFmpeg、Silero ONNX 模型与内置 Profile 均可用（不要求配置 API Key）",
    )

    return parser


def run_providers(defaults_path: Path) -> int:
    defaults = load_engine_config(defaults_path)
    providers_cfg = get_providers_config(defaults)
    tr_cfg = defaults.get("transcript") or {}
    tl_cfg = defaults.get("translate") or {}

    print("=== [AutoSub 已配置厂商与路由] ===", flush=True)
    for name, cfg in providers_cfg.items():
        proto = str(cfg.get("protocol") or name)
        caps = provider_capabilities(proto)
        api_base = str(cfg.get("api_base") or "") or "(未设置)"
        api_mode = str(cfg.get("api_mode") or "")
        mode_part = f" | mode={api_mode}" if api_mode else ""
        key_ok = bool(resolve_provider_api_key(cfg))
        env_hint = " / ".join(cfg.get("api_key_envs") or ()) or f"{name.upper()}_API_KEY"
        key_status = "已配置" if key_ok else f"未配置 ({env_hint})"
        print(f"  - {name} [protocol={proto}, 能力={caps}{mode_part}]", flush=True)
        print(f"      api_base: {api_base} | API Key: {key_status}", flush=True)

    tr_provider = str(tr_cfg.get("provider") or "gemini")
    tr_model = str(tr_cfg.get("model") or "gemini-3.5-transcribe")
    ref_enabled = bool(tr_cfg.get("ref_enabled", False))
    ref_model = str(tr_cfg.get("ref_model") or "").strip()
    ref_provider = str(tr_cfg.get("ref_provider") or "").strip()
    ref_label = f"{ref_provider}/{ref_model}" if ref_provider and ref_model else (ref_model or ref_provider or "(未指定)")
    tl_specs = normalize_translate_model_specs(
        tl_cfg.get("models"),
        default_provider=str(tl_cfg.get("provider") or "gemini"),
        known_providers=providers_cfg,
    )
    tl_desc = " -> ".join(s.display_label for s in tl_specs)
    print(f"默认转录路由: {tr_provider}/{tr_model}", flush=True)
    print(f"双 ASR 参考轨: {'已启用 (' + ref_label + ')' if ref_enabled else '未启用 (可在 config/defaults.yaml 中开启 transcript.ref_enabled)'}", flush=True)
    print(f"默认翻译链: {tl_desc}", flush=True)
    return 0


def run_doctor(defaults_path: Path, *, strict_bundle: bool = False) -> int:
    defaults = load_engine_config(defaults_path)
    tools_cfg = defaults.get("tools") or {}
    providers_cfg = get_providers_config(defaults)

    ffmpeg_resolved = resolve_binary(str(tools_cfg.get("ffmpeg_bin") or ""), ["ffmpeg.exe", "ffmpeg"])
    ffmpeg_ok = Path(ffmpeg_resolved).is_file() or shutil.which(ffmpeg_resolved) is not None

    silero_ok, silero_backend, onnx_path, ort_lib = probe_silero_vad_runtime()

    profiles = list_profiles()
    profiles_ok = len(profiles) > 0
    gemini_ok = bool(resolve_provider_api_key(providers_cfg["gemini"]))

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
    print(f"Gemini API Key: {'已配置' if gemini_ok else '未配置（请设置 GEMINI_API_KEY 或创建 .env）'}", flush=True)
    print("多厂商 API Key 状态:", flush=True)
    for name, cfg in providers_cfg.items():
        if name == "gemini":
            continue
        proto = str(cfg.get("protocol") or name)
        caps = provider_capabilities(proto)
        key_ok = bool(resolve_provider_api_key(cfg))
        env_hint = " / ".join(cfg.get("api_key_envs") or ()) or f"{name.upper()}_API_KEY"
        status = "已配置" if key_ok else f"未配置（可选，环境变量 {env_hint}）"
        print(f"  - {name} ({caps}): {status}", flush=True)

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

    if args.command == "providers":
        return run_providers(defaults_path)

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
            ref_output_path=Path(args.ref_output).resolve() if args.ref_output else None,
            vad_override=args.vad,
            provider_override=args.provider,
            model_override=args.model,
            ref_enabled_override=args.ref_enabled,
            ref_provider_override=args.ref_provider,
            ref_model_override=args.ref_model,
            max_chunk_seconds_override=args.max_chunk_seconds,
            language_override=args.lang,
            secondary_languages_override=args.secondary_lang or None,
            keyterms_override=args.keyterm or None,
            diarize_override=args.diarize,
            no_verbatim_override=args.no_verbatim,
            numbers_format_override=args.numbers_format,
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
            provider_override=args.provider,
            model_override=args.model,
            reasoning_effort_override=args.reasoning_effort,
            auto_context_override=args.auto_context,
            merge_cues_override=args.merge_cues,
            polish_timing_override=args.polish_timing,
            ref_srt_path=Path(args.ref_srt).resolve() if args.ref_srt else None,
            output_path=Path(args.output).resolve() if args.output else None,
        )

    if args.command == "run":
        try:
            raw_srt, ref_srt = transcribe_media_pair(
                Path(args.input).resolve(),
                defaults_path=defaults_path,
                profile_name=args.profile,
                output_path=Path(args.srt_output).resolve() if args.srt_output else None,
                ref_output_path=Path(args.ref_srt_output).resolve() if args.ref_srt_output else None,
                vad_override=args.vad,
                provider_override=args.transcript_provider,
                model_override=args.transcript_model,
                ref_enabled_override=args.ref_enabled,
                ref_provider_override=args.ref_provider,
                ref_model_override=args.ref_model,
                max_chunk_seconds_override=args.max_chunk_seconds,
                language_override=args.lang,
                secondary_languages_override=args.secondary_lang or None,
                keyterms_override=args.keyterm or None,
                diarize_override=args.diarize,
                no_verbatim_override=args.no_verbatim,
                numbers_format_override=args.numbers_format,
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
                provider_override=args.translate_provider,
                model_override=args.translate_model,
                reasoning_effort_override=args.reasoning_effort,
                auto_context_override=args.auto_context,
                merge_cues_override=args.merge_cues,
                polish_timing_override=args.polish_timing,
                ref_srt_path=ref_srt,
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
