"""Comprehensive offline unit tests for AutoSub (zero live network calls)."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import yaml

import main as autosub_main
from src.config import (
    PROJECT_ROOT,
    get_providers_config,
    load_engine_config,
    resolve_api_key,
    resolve_binary,
    resolve_provider_api_key,
)
from src.http_client import (
    GeminiApiError,
    ProviderApiError,
    build_multipart_payload,
    post_json,
    post_multipart,
    upload_resumable_file,
)
from src.profile import (
    list_profiles,
    load_glossary,
    load_profile,
    merge_glossaries,
    parse_var_overrides,
    render_glossary_section,
    render_prompt,
)
from src.providers import (
    ModelSpec,
    ProviderSettings,
    build_provider_settings,
    create_transcribe_provider,
    create_translate_provider,
    parse_model_spec,
)
from src.providers.elevenlabs import (
    ElevenLabsTranscribeProvider,
    build_elevenlabs_form_fields,
    extract_elevenlabs_words,
    sanitize_keyterms,
)
from src.providers.openai import (
    OpenAICompatTranscribeProvider,
    OpenAITranslateProvider,
    extract_whisper_verbose_words,
)
from src.providers.openai_compat import OpenAICompatTranslateProvider
from src.srt import (
    SrtCue,
    align_reference_srt,
    append_srt_cues,
    clamp_stretched_words,
    format_srt_ts,
    group_words_to_cues,
    map_word_to_original,
    parse_srt,
    parse_srt_ts,
    polish_srt_timing,
    write_srt,
)
from src.transcribe import (
    call_gemini_transcribe,
    extract_words_from_interaction,
    map_transcribed_words_to_original,
    resolve_dual_asr_spec,
    transcribe_media_pair,
)
from src.translate import (
    DEFAULT_TRANSLATE_MODELS,
    SubtitleTranslator,
    _parse_compact_translation,
    _parse_cue_entries,
    _resolve_translate_models_and_providers,
    format_cue_lines,
    normalize_translate_chain,
    normalize_translate_model_specs,
    split_cues_evenly,
)
from src.vad import (
    find_onnxruntime_library,
    find_silero_onnx_model,
    pack_segments_into_chunks,
    probe_silero_vad_runtime,
)


def _mk_resp(payload_obj: dict, *, tier: str = "STANDARD", in_tok: int = 120, out_tok: int = 45) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps(payload_obj, ensure_ascii=False)}]}}],
        "usageMetadata": {
            "promptTokenCount": in_tok,
            "candidatesTokenCount": out_tok,
            "serviceTier": tier,
        },
    }


class TestConfigAndProfiles(unittest.TestCase):
    def test_engine_defaults_and_chain_has_no_flex(self) -> None:
        cfg = load_engine_config()
        chain = normalize_translate_chain(cfg["translate"]["models"])
        self.assertEqual(
            chain,
            [
                ("gemini-4-hydrogen", "standard"),
                ("gemini-3.8-flash", "standard"),
                ("gemini-3.5-flash-lite", "standard"),
            ],
        )
        self.assertEqual(normalize_translate_chain(DEFAULT_TRANSLATE_MODELS), chain)

    def test_resolve_api_key_priority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".env").write_text('GEMINI_API_KEY="from_dotenv"\n', encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                with mock.patch("pathlib.Path.home", side_effect=RuntimeError("no home")):
                    self.assertEqual(resolve_api_key("explicit", project_root=root), "explicit")
                    self.assertEqual(resolve_api_key("", project_root=root), "from_dotenv")
            with mock.patch.dict("os.environ", {"GEMINI_API_KEY": "from_env"}, clear=True):
                self.assertEqual(resolve_api_key("", project_root=root), "from_env")

    def test_builtin_profiles_list_and_load(self) -> None:
        names = [n for n, _, _ in list_profiles()]
        self.assertIn("default", names)
        self.assertIn("twitch_dota2", names)

        default_p = load_profile("default")
        self.assertEqual(default_p.glossaries, [])
        prompt_default = render_prompt(
            default_p,
            vars_override={"topic": "Distributed Systems", "speaker": "Alice"},
            glossary_text="",
        )
        self.assertIn("Distributed Systems", prompt_default)
        self.assertIn("Alice", prompt_default)
        self.assertNotIn("Terminology Glossary", prompt_default)
        self.assertIn('"keep": [', prompt_default)

        dota_p = load_profile("twitch_dota2")
        self.assertEqual(len(dota_p.glossaries), 1)
        self.assertTrue(dota_p.glossaries[0].is_file())
        g_text = render_glossary_section(merge_glossaries(dota_p.glossaries))
        prompt_dota = render_prompt(
            dota_p,
            vars_override={"streamer": "4oapm", "hero": "Visage / 死灵龙"},
            glossary_text=g_text,
        )
        self.assertIn("4oapm", prompt_dota)
        self.assertIn("Visage / 死灵龙", prompt_dota)
        self.assertIn("### Terminology Glossary", prompt_dota)
        self.assertIn("死灵龙", prompt_dota)

    def test_external_profile_and_multi_glossary_merge(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            g1 = d / "g1.yaml"
            g2 = d / "g2.yaml"
            g1.write_text("frameworks:\n  PyTorch: [torch, py-torch]\n", encoding="utf-8")
            g2.write_text(
                "frameworks:\n  PyTorch: [py-torch, pt]\nconcepts:\n  量化: [quantization]\n",
                encoding="utf-8",
            )
            prof = d / "tech_talk.yaml"
            prof.write_text(
                yaml.safe_dump(
                    {
                        "name": "tech_talk",
                        "target_language": "Japanese",
                        "bilingual": False,
                        "bilingual_order": ["source", "target"],
                        "translate": {
                            "glossaries": ["g1.yaml", str(g2)],
                            "default_vars": {"event": "KubeCon"},
                            "prompt_template": "Event={event} Lang={target_language}\n{glossary_section}\n{{}}",
                        },
                    },
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            loaded = load_profile(prof)
            self.assertFalse(loaded.bilingual)
            self.assertEqual(loaded.bilingual_order, ["source", "target"])
            merged = merge_glossaries(loaded.glossaries)
            self.assertEqual(merged["frameworks"]["PyTorch"], ["torch", "py-torch", "pt"])
            self.assertEqual(merged["concepts"]["量化"], ["quantization"])
            rendered = render_prompt(loaded, glossary_text=render_glossary_section(merged))
            self.assertIn("Event=KubeCon Lang=Japanese", rendered)
            self.assertIn("torch/py-torch/pt=PyTorch", rendered)
            self.assertIn("quantization=量化", rendered)
            self.assertIn("{}", rendered)

    def test_dota_glossary_has_126_heroes_and_refined_creeps(self) -> None:
        g = load_glossary(PROJECT_ROOT / "glossaries" / "dota2_ru_en_zh.yaml")
        self.assertEqual(len(g["hero_aliases_ru_en"]), 126)
        self.assertIn("线上小兵", g["mechanics_and_slang"])
        self.assertIn("中立野怪", g["mechanics_and_slang"])
        self.assertIn("远古野", g["mechanics_and_slang"])
        self.assertIn("野点", g["mechanics_and_slang"])

    def test_parse_var_overrides(self) -> None:
        self.assertEqual(
            parse_var_overrides(["hero=Visage / 死灵龙", "streamer = 4oapm", "note=a=b"]),
            {"hero": "Visage / 死灵龙", "streamer": "4oapm", "note": "a=b"},
        )
        with self.assertRaises(ValueError):
            parse_var_overrides(["invalid_no_equals"])
        with self.assertRaises(ValueError):
            parse_var_overrides(["=empty_key"])


class TestSrtAndVad(unittest.TestCase):
    def test_srt_roundtrip_and_incremental_append(self) -> None:
        self.assertEqual(format_srt_ts(3661.234), "01:01:01,234")
        self.assertAlmostEqual(parse_srt_ts("01:01:01,234"), 3661.234, places=3)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "demo.srt"
            write_srt([(1.0, 2.5, ["Line A"]), (3.0, 4.0, ["Line B"])], p)
            cues = parse_srt(p)
            self.assertEqual(len(cues), 2)
            self.assertEqual(cues[0].text, "Line A")

            p2 = Path(td) / "stream.srt"
            n1 = append_srt_cues([(0.0, 1.0, ["甲", "A"])], p2, start_index=1, truncate=True)
            n2 = append_srt_cues([(1.5, 2.5, ["乙", "B"])], p2, start_index=n1 + 1, truncate=False)
            self.assertEqual(n1, 1)
            self.assertEqual(n2, 2)
            text = p2.read_text(encoding="utf-8")
            self.assertIn("1\n00:00:00,000 --> 00:00:01,000\n甲\nA", text)
            self.assertIn("2\n00:00:01,500 --> 00:00:02,500\n乙\nB", text)

    def test_pack_segments_map_and_group_words(self) -> None:
        chunks = pack_segments_into_chunks([(10.0, 20.0), (50.0, 65.0), (100.0, 120.0)], max_chunk_seconds=30.0, silence_gap_s=0.5)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 2)
        self.assertEqual(len(chunks[1]), 1)

        seg_idx, os_, oe_ = map_word_to_original(11.5, 12.2, chunks[0])
        self.assertEqual(seg_idx, 1)
        self.assertAlmostEqual(os_, 51.0, places=2)
        self.assertAlmostEqual(oe_, 51.7, places=2)

        resp = {
            "steps": [{
                "content": [{
                    "annotations": [
                        {"type": "word_info", "text": "Hello", "start_offset": "0.5s", "end_offset": "0.9s"},
                        {"type": "word_info", "text": "world.", "start_offset": "1.0s", "end_offset": "1.5s"},
                        {"type": "word_info", "text": "Next", "start_offset": "11.0s", "end_offset": "11.4s"},
                    ]
                }]
            }]
        }
        words = extract_words_from_interaction(resp, chunks[0])
        self.assertEqual(len(words), 3)
        cues = group_words_to_cues(words)
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].text, "Hello world.")
        self.assertEqual(cues[1].text, "Next")

    def test_transcribe_retries_on_503_and_429(self) -> None:
        sleeps: list[float] = []
        side_effects = [
            GeminiApiError("service_unavailable", "high demand", http_status=503),
            GeminiApiError("too_many_requests", "Please retry in 5s", http_status=429),
            {"steps": []},
        ]
        idx = 0

        def fake_post(url: str, payload: dict, **_kw) -> dict:
            nonlocal idx
            item = side_effects[idx]
            idx += 1
            if isinstance(item, Exception):
                raise item
            return item

        out = call_gemini_transcribe(
            "https://example.com/files/abc",
            api_key="test",
            max_retries=5,
            sleep_fn=sleeps.append,
            post_fn=fake_post,
        )
        self.assertEqual(out, {"steps": []})
        self.assertEqual(sleeps, [8, 8])


class TestTranslator(unittest.TestCase):
    def test_split_cues_evenly(self) -> None:
        cues = [SrtCue(i, float(i), float(i + 1), f"t{i}") for i in range(1, 533)]
        batches = split_cues_evenly(cues, max_per_batch=280)
        self.assertEqual([len(b) for b in batches], [266, 266])

    def test_format_cue_lines_bilingual_and_mono(self) -> None:
        self.assertEqual(format_cue_lines("中文", "EN", bilingual=True, bilingual_order=("target", "source")), ["中文", "EN"])
        self.assertEqual(format_cue_lines("中文", "EN", bilingual=True, bilingual_order=("source", "target")), ["EN", "中文"])
        self.assertEqual(format_cue_lines("中文", "EN", bilingual=False), ["中文"])

    def test_parse_compact_with_guardrails_and_missing_id_safety(self) -> None:
        batch = [
            SrtCue(1, 0.0, 1.0, "О."),
            SrtCue(2, 1.0, 2.0, "Я бы взял БКБ."),
            SrtCue(3, 2.0, 3.0, "Короткий текст."),
            SrtCue(4, 3.0, 4.0, "Missing from model output"),
        ]
        parsed = {
            "drop": [1],
            "keep": [
                [2, "我会出BKB。"],
                [3, "短句。", "A wildly bloated merged sentence that is far longer than the original cue text and must be rejected."],
            ],
        }
        res, kept, dropped = _parse_compact_translation(parsed, batch)
        self.assertEqual((kept, dropped), (3, 1))
        self.assertTrue(res[1]["drop"])
        self.assertEqual(res[2]["target"], "我会出BKB。")
        self.assertEqual(res[2]["src"], "Я бы взял БКБ.")
        # Bloated corrected_src rejected -> original preserved
        self.assertEqual(res[3]["src"], "Короткий текст.")
        # Missing ID preserved safely
        self.assertFalse(res[4]["drop"])
        self.assertEqual(res[4]["target"], "Missing from model output")

    def test_503_retries_three_times_then_switches_model(self) -> None:
        calls: list[str] = []
        sleeps: list[float] = []

        def fake_post(url: str, payload: dict, **_kw) -> dict:
            calls.append(url.split("/models/")[1].split(":")[0])
            if calls[-1] == "gemini-4-hydrogen":
                raise GeminiApiError("503", "high demand", http_status=503)
            return _mk_resp({"keep": [[1, "翻译结果"]], "drop": []})

        tr = SubtitleTranslator(api_key="k", retry_503=3, post_fn=fake_post, sleep_fn=sleeps.append)
        out = tr.translate_cues([SrtCue(1, 0.0, 1.0, "hi")], system_prompt="sys")
        self.assertEqual(calls, ["gemini-4-hydrogen", "gemini-4-hydrogen", "gemini-4-hydrogen", "gemini-3.8-flash"])
        self.assertEqual(sleeps, [5.0, 10.0])
        self.assertEqual(out[0][2], ["翻译结果", "hi"])

    def test_429_daily_rpd_skips_model_for_remaining_batches(self) -> None:
        calls: list[str] = []

        def fake_post(url: str, payload: dict, **_kw) -> dict:
            model = url.split("/models/")[1].split(":")[0]
            calls.append(model)
            if model == "gemini-4-hydrogen":
                raise GeminiApiError("429", "Quota exceeded for RequestsPerDay", http_status=429)
            body = json.loads(payload["contents"][0]["parts"][0]["text"])
            kept = [[cid, f"T{cid}"] for cid, _ in body["subtitles"]]
            return _mk_resp({"keep": kept, "drop": []})

        cues = [SrtCue(1, 0.0, 1.0, "a"), SrtCue(2, 1.0, 2.0, "b")]
        tr = SubtitleTranslator(api_key="k", retry_503=3, post_fn=fake_post, sleep_fn=lambda _: None)
        out = tr.translate_cues(cues, system_prompt="sys", max_cues_per_batch=1)
        self.assertEqual(len(out), 2)
        # Batch 1 tries hydrogen -> 429 RPD -> switches to 3.8-flash; Batch 2 skips hydrogen directly
        self.assertEqual(calls, ["gemini-4-hydrogen", "gemini-3.8-flash", "gemini-3.8-flash"])

    def test_incremental_write_persists_completed_batches_on_midrun_failure(self) -> None:
        batch_num = 0

        def fake_post(url: str, payload: dict, **_kw) -> dict:
            nonlocal batch_num
            batch_num += 1
            if batch_num == 1:
                return _mk_resp({"keep": [[1, "第一批译文"]], "drop": []})
            raise GeminiApiError("503", "overloaded", http_status=503)

        cues = [SrtCue(1, 0.0, 1.0, "one"), SrtCue(2, 2.0, 3.0, "two")]
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "out.srt"
            tr = SubtitleTranslator(
                api_key="k",
                models=[{"model": "gemini-4-hydrogen", "service_tier": "standard"}],
                retry_503=1,
                post_fn=fake_post,
                sleep_fn=lambda _: None,
            )
            with self.assertRaises(RuntimeError):
                tr.translate_cues(cues, system_prompt="sys", max_cues_per_batch=1, output_path=out_path)
            saved = out_path.read_text(encoding="utf-8")
            self.assertIn("第一批译文", saved)
            self.assertIn("one", saved)


class TestCliParser(unittest.TestCase):
    def test_parse_all_subcommands(self) -> None:
        parser = autosub_main.build_parser()

        a_tr = parser.parse_args(["transcript", "demo.mp4", "--profile", "twitch_dota2", "--vad", "silencedetect", "-o", "out.srt"])
        self.assertEqual(a_tr.command, "transcript")
        self.assertEqual(a_tr.profile, "twitch_dota2")
        self.assertEqual(a_tr.vad, "silencedetect")

        a_tl = parser.parse_args([
            "translate", "demo.srt", "-p", "twitch_dota2",
            "--var", "hero=死灵龙", "--var", "streamer=4oapm",
            "--mono", "--batch-size", "200",
        ])
        self.assertEqual(a_tl.command, "translate")
        self.assertEqual(a_tl.var, ["hero=死灵龙", "streamer=4oapm"])
        self.assertIs(a_tl.bilingual, False)
        self.assertEqual(a_tl.batch_size, 200)

        a_run = parser.parse_args([
            "run", "vod.mp4", "-p", "default",
            "--var", "topic=AI", "--bilingual", "--srt-output", "raw.srt", "-o", "bi.srt",
        ])
        self.assertEqual(a_run.command, "run")
        self.assertIs(a_run.bilingual, True)
        self.assertEqual(a_run.srt_output, "raw.srt")
        self.assertEqual(a_run.output, "bi.srt")

        a_prof = parser.parse_args(["profiles"])
        self.assertEqual(a_prof.command, "profiles")

        a_doc = parser.parse_args(["doctor", "--strict-bundle"])
        self.assertEqual(a_doc.command, "doctor")
        self.assertTrue(a_doc.strict_bundle)


class TestWindowsBundleAndHttpClient(unittest.TestCase):
    def test_resolve_binary_prefers_bundled_bin_and_silero_finds_bundled_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app_root = Path(td) / "app"
            bundle_root = app_root / "_internal"
            (app_root / "bin").mkdir(parents=True)
            (bundle_root / "models").mkdir(parents=True)
            (bundle_root / "lib").mkdir(parents=True)
            fake_ffmpeg = app_root / "bin" / "ffmpeg.exe"
            fake_ffmpeg.write_bytes(b"MZ")
            fake_onnx = bundle_root / "models" / "silero_vad.onnx"
            fake_onnx.write_bytes(b"ONNX")
            fake_ort = bundle_root / "lib" / ("onnxruntime.dll" if Path("C:/").drive or __import__("os").name == "nt" else "libonnxruntime.so")
            fake_ort.write_bytes(b"ORT")

            resolved = resolve_binary(
                "",
                ["ffmpeg.exe", "ffmpeg"],
                app_root=app_root,
                bundle_root=bundle_root,
            )
            self.assertEqual(Path(resolved), fake_ffmpeg)

            with mock.patch.dict("os.environ", {}, clear=True):
                found_onnx = find_silero_onnx_model(app_root=app_root, bundle_root=bundle_root)
                found_ort = find_onnxruntime_library(app_root=app_root, bundle_root=bundle_root)
                ok, backend, m_path, lib_path = probe_silero_vad_runtime(app_root=app_root, bundle_root=bundle_root)
            self.assertEqual(found_onnx, fake_onnx)
            self.assertEqual(found_ort, fake_ort)
            self.assertTrue(ok)
            self.assertEqual(backend, "ctypes_capi")
            self.assertEqual(m_path, fake_onnx)
            self.assertEqual(lib_path, fake_ort)

    def test_resolve_api_key_falls_back_to_user_config_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app_root = Path(td) / "app"
            user_cfg = Path(td) / "appdata"
            app_root.mkdir()
            user_cfg.mkdir()
            (user_cfg / ".env").write_text("GEMINI_API_KEY=from_appdata\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                self.assertEqual(
                    resolve_api_key("", project_root=app_root, user_config_dir=user_cfg),
                    "from_appdata",
                )

    def test_pure_python_http_client_post_and_resumable_upload(self) -> None:
        class FakeResp:
            def __init__(self, body: bytes, headers: dict[str, str] | None = None):
                self._body = body
                self.headers = headers or {}

            def read(self) -> bytes:
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        # 1. post_json success + HTTP 429 error parsing
        fake_opener = mock.MagicMock()
        fake_opener.open.return_value = FakeResp(json.dumps({"ok": True}).encode())
        with mock.patch("src.http_client._build_opener", return_value=fake_opener):
            self.assertEqual(post_json("https://example.com/x", {"a": 1}, api_key="k"), {"ok": True})

        err_body = io.BytesIO(json.dumps({"error": {"code": 429, "message": "Quota exceeded per day"}}).encode())
        http_err = urllib.error.HTTPError("https://example.com/x", 429, "Too Many Requests", {}, err_body)
        fake_opener.open.side_effect = http_err
        with mock.patch("src.http_client._build_opener", return_value=fake_opener):
            with self.assertRaises(GeminiApiError) as ctx:
                post_json("https://example.com/x", {"a": 1}, api_key="k")
        self.assertTrue(ctx.exception.is_429_rate_limit)
        self.assertTrue(ctx.exception.is_daily_quota_exhausted)

        # 2. upload_resumable_file 2-step flow
        with tempfile.TemporaryDirectory() as td:
            mp3 = Path(td) / "chunk.mp3"
            mp3.write_bytes(b"ID3fakeaudio")
            upload_opener = mock.MagicMock()
            upload_opener.open.side_effect = [
                FakeResp(b"", headers={"x-goog-upload-url": "https://upload.example.com/session1"}),
                FakeResp(json.dumps({"file": {"name": "files/123", "uri": "https://example.com/files/123"}}).encode()),
            ]
            with mock.patch("src.http_client._build_opener", return_value=upload_opener):
                finfo = upload_resumable_file(mp3, api_key="k")
            self.assertEqual(finfo["name"], "files/123")
            self.assertEqual(finfo["uri"], "https://example.com/files/123")

    def test_windows_packaging_files_exist_and_workflow_valid_yaml(self) -> None:
        self.assertTrue((PROJECT_ROOT / "packaging" / "windows" / "autosub.spec").is_file())
        self.assertTrue((PROJECT_ROOT / "packaging" / "windows" / "autosub.iss").is_file())
        self.assertTrue((PROJECT_ROOT / "packaging" / "windows" / "build_windows.ps1").is_file())
        wf_path = PROJECT_ROOT / ".github" / "workflows" / "build-windows.yml"
        self.assertTrue(wf_path.is_file())
        wf = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        self.assertIn("build-windows", wf["jobs"])


class TestMultiProviderArchitecture(unittest.TestCase):
    def test_providers_config_merges_legacy_api_and_custom_aliases(self) -> None:
        cfg = {
            "api": {
                "api_key": "legacy-gem-key",
                "api_base": "https://custom-gemini.example/v1beta",
                "proxy": "http://127.0.0.1:7890",
                "timeout_seconds": 240,
            },
            "providers": {
                "DeepSeek": {
                    "protocol": "openai_compatible",
                    "api_key_envs": ["DEEPSEEK_API_KEY"],
                    "api_base": "https://api.deepseek.com/v1",
                },
            },
        }
        providers = get_providers_config(cfg)
        self.assertEqual(providers["gemini"]["api_key"], "legacy-gem-key")
        self.assertEqual(providers["gemini"]["api_base"], "https://custom-gemini.example/v1beta")
        self.assertEqual(providers["gemini"]["proxy"], "http://127.0.0.1:7890")
        self.assertEqual(providers["gemini"]["timeout_seconds"], 240)
        self.assertEqual(providers["elevenlabs"]["protocol"], "elevenlabs")
        self.assertEqual(providers["openai"]["api_mode"], "responses")
        self.assertIn("deepseek", providers)
        self.assertEqual(providers["deepseek"]["protocol"], "openai_compatible")
        self.assertEqual(providers["deepseek"]["api_mode"], "chat_completions")
        self.assertEqual(providers["deepseek"]["proxy"], "http://127.0.0.1:7890")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".env").write_text("DEEPSEEK_API_KEY=ds-from-dotenv\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {}, clear=True):
                with mock.patch("pathlib.Path.home", side_effect=RuntimeError("no home")):
                    resolved = resolve_provider_api_key(providers["deepseek"], project_root=root)
            self.assertEqual(resolved, "ds-from-dotenv")

    def test_parse_model_spec_and_provider_factory_routing(self) -> None:
        s1 = parse_model_spec("elevenlabs/scribe_v3")
        self.assertEqual((s1.provider, s1.model), ("elevenlabs", "scribe_v3"))

        s2 = parse_model_spec("scribe_v3")
        self.assertEqual((s2.provider, s2.model), ("elevenlabs", "scribe_v3"))

        s3 = parse_model_spec("openai/gpt-6.2-sol", reasoning_effort_override="medium")
        self.assertEqual((s3.provider, s3.model, s3.reasoning_effort), ("openai", "gpt-6.2-sol", "medium"))

        s4 = parse_model_spec("gpt-6.2-sol")
        self.assertEqual((s4.provider, s4.model), ("openai", "gpt-6.2-sol"))

        s5 = parse_model_spec("models/gemini-3.8-flash")
        self.assertEqual((s5.provider, s5.model), ("gemini", "gemini-3.8-flash"))

        # Explicit custom provider preserves vendor/model slash (e.g. OpenRouter)
        s6 = parse_model_spec(
            {"provider": "openrouter", "model": "openai/gpt-6.2-sol"},
            known_providers={"openrouter": {"protocol": "openai_compatible"}},
        )
        self.assertEqual((s6.provider, s6.model), ("openrouter", "openai/gpt-6.2-sol"))

        # ElevenLabs is STT-only -> rejected for translation
        el_settings = ProviderSettings(
            name="elevenlabs",
            protocol="elevenlabs",
            api_key="el-key",
            api_base="https://api.elevenlabs.io/v1",
        )
        self.assertIsInstance(create_transcribe_provider(el_settings), ElevenLabsTranscribeProvider)
        with self.assertRaises(ValueError):
            create_translate_provider(el_settings)

    def test_multipart_builder_and_provider_error_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mp3 = Path(td) / "sample.mp3"
            mp3.write_bytes(b"ID3-FAKE-AUDIO")
            ctype, body = build_multipart_payload(
                fields=[
                    ("model_id", "scribe_v3"),
                    ("secondary_language_codes", "eng"),
                    ("secondary_language_codes", "jpn"),
                ],
                files=[("file", mp3, "audio/mpeg")],
                boundary="TESTBOUNDARY",
            )
            self.assertEqual(ctype, "multipart/form-data; boundary=TESTBOUNDARY")
            self.assertIn(b'name="secondary_language_codes"\r\n\r\neng', body)
            self.assertIn(b'name="secondary_language_codes"\r\n\r\njpn', body)
            self.assertIn(b'filename="sample.mp3"\r\nContent-Type: audio/mpeg\r\n\r\nID3-FAKE-AUDIO', body)

            # ElevenLabs 402 detail dict error parsing
            fake_opener = mock.MagicMock()
            el_err_body = io.BytesIO(
                json.dumps({"detail": {"status": "quota_exceeded", "message": "Not enough credits"}}).encode()
            )
            fake_opener.open.side_effect = urllib.error.HTTPError(
                "https://api.elevenlabs.io/v1/speech-to-text", 402, "Payment Required", {}, el_err_body
            )
            with mock.patch("src.http_client._build_opener", return_value=fake_opener):
                with self.assertRaises(ProviderApiError) as ctx:
                    post_multipart(
                        "https://api.elevenlabs.io/v1/speech-to-text",
                        fields=[("model_id", "scribe_v3")],
                        files=[("file", mp3, "audio/mpeg")],
                        headers={"xi-api-key": "k"},
                        provider_name="ElevenLabs",
                    )
            self.assertEqual(ctx.exception.http_status, 402)
            self.assertTrue(ctx.exception.is_billing_or_quota_exhausted)
            self.assertIn("Not enough credits", ctx.exception.message)

        # OpenAI 429 with Retry-After header
        fake_opener2 = mock.MagicMock()
        oa_err_body = io.BytesIO(
            json.dumps({"error": {"type": "rate_limit_exceeded", "code": "rate_limit_exceeded", "message": "RPM limit"}}).encode()
        )
        fake_opener2.open.side_effect = urllib.error.HTTPError(
            "https://api.openai.com/v1/responses", 429, "Too Many Requests", {"Retry-After": "7"}, oa_err_body
        )
        with mock.patch("src.http_client._build_opener", return_value=fake_opener2):
            with self.assertRaises(ProviderApiError) as ctx2:
                post_json("https://api.openai.com/v1/responses", {"model": "gpt-6.2-sol"}, headers={"Authorization": "Bearer k"})
        self.assertTrue(ctx2.exception.is_429_rate_limit)
        self.assertEqual(ctx2.exception.retry_after_seconds, 7)

    def test_elevenlabs_form_fields_v3_vs_v2_and_keyterm_sanitization(self) -> None:
        cleaned = sanitize_keyterms([
            "Visage",
            "Visage",
            "Black King Bar",
            "too many words in one single keyterm entry here",
            "x" * 51,
            "  Shadow   Fiend \n",
        ])
        self.assertEqual(cleaned, ["Visage", "Black King Bar", "Shadow Fiend"])

        v3_fields = build_elevenlabs_form_fields(
            model="scribe_v3",
            options={
                "language_code": "rus",
                "secondary_language_codes": ["eng", "rus", "eng", "deu"],
                "numbers_format": "digits",
                "no_verbatim": True,
                "diarize": True,
                "num_speakers": 2,
                "keyterms": ["Visage", "Roshan"],
            },
        )
        self.assertIn(("model_id", "scribe_v3"), v3_fields)
        self.assertIn(("timestamps_granularity", "word"), v3_fields)
        self.assertIn(("tag_audio_events", "false"), v3_fields)
        self.assertEqual([v for k, v in v3_fields if k == "secondary_language_codes"], ["eng", "deu"])
        self.assertIn(("numbers_format", "digits"), v3_fields)
        self.assertIn(("no_verbatim", "true"), v3_fields)
        self.assertIn(("diarize", "true"), v3_fields)
        self.assertIn(("num_speakers", "2"), v3_fields)
        self.assertEqual([v for k, v in v3_fields if k == "keyterms"], ["Visage", "Roshan"])

        # v2 must NOT receive v3-only fields (`secondary_language_codes`, `numbers_format`), and empty keyterms omitted
        v2_fields = build_elevenlabs_form_fields(
            model="scribe_v2",
            options={
                "language_code": "rus",
                "secondary_language_codes": ["eng"],
                "numbers_format": "digits",
                "keyterms": [],
            },
        )
        keys_v2 = [k for k, _ in v2_fields]
        self.assertNotIn("secondary_language_codes", keys_v2)
        self.assertNotIn("numbers_format", keys_v2)
        self.assertNotIn("keyterms", keys_v2)

    def test_elevenlabs_words_spacing_cjk_diarize_and_retries(self) -> None:
        # 1. Latin/Russian spacing + skipping audio_event + speaker change splitting
        resp_ru = {
            "words": [
                {"text": "Привет", "start": 0.1, "end": 0.4, "type": "word", "speaker_id": "speaker_0"},
                {"text": " ", "start": 0.4, "end": 0.45, "type": "spacing"},
                {"text": "BKB", "start": 0.45, "end": 0.8, "type": "word", "speaker_id": "speaker_0"},
                {"text": "(laughter)", "start": 0.8, "end": 0.9, "type": "audio_event"},
                {"text": " ", "start": 0.9, "end": 0.95, "type": "spacing"},
                {"text": "Да!", "start": 1.0, "end": 1.3, "type": "word", "speaker_id": "speaker_1"},
            ]
        }
        words_ru = extract_elevenlabs_words(resp_ru)
        self.assertEqual([w.text for w in words_ru], ["Привет", "BKB", "Да!"])
        self.assertEqual([w.space_before for w in words_ru], [False, True, True])
        chunks = pack_segments_into_chunks([(10.0, 20.0)], max_chunk_seconds=60.0)
        mapped_ru = map_transcribed_words_to_original(words_ru, chunks[0])
        cues_ru = group_words_to_cues(mapped_ru)
        self.assertEqual(len(cues_ru), 2)
        self.assertEqual(cues_ru[0].text, "Привет BKB")
        self.assertEqual(cues_ru[1].text, "Да!")

        # 2. CJK words without `spacing` tokens join cleanly without spurious spaces
        resp_zh = {
            "words": [
                {"text": "今天", "start": 0.0, "end": 0.3, "type": "word"},
                {"text": "天气", "start": 0.3, "end": 0.6, "type": "word"},
                {"text": "很好。", "start": 0.6, "end": 0.9, "type": "word"},
            ]
        }
        words_zh = extract_elevenlabs_words(resp_zh)
        mapped_zh = map_transcribed_words_to_original(words_zh, chunks[0])
        cues_zh = group_words_to_cues(mapped_zh)
        self.assertEqual(len(cues_zh), 1)
        self.assertEqual(cues_zh[0].text, "今天天气很好。")

        # 3. Retries on 429 and fast-fails on 402 credits / 401 unusual activity
        settings = ProviderSettings(
            name="elevenlabs",
            protocol="elevenlabs",
            api_key="el-test",
            api_base="https://api.elevenlabs.io/v1",
        )
        sleeps: list[float] = []
        calls = 0

        def fake_mp(url: str, **_kw) -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ProviderApiError("429", "too many concurrent", http_status=429, retry_after=4)
            return {"language_code": "rus", "language_probability": 0.96, "audio_duration_secs": 12.5, "words": resp_ru["words"]}

        provider = ElevenLabsTranscribeProvider(settings, multipart_fn=fake_mp)
        with tempfile.TemporaryDirectory() as td:
            mp3 = Path(td) / "c.mp3"
            mp3.write_bytes(b"MP3")
            res = provider.transcribe_chunk(mp3, model="scribe_v3", max_retries=3, sleep_fn=sleeps.append)
            self.assertEqual(len(res.words), 3)
            self.assertEqual(sleeps, [6])
            self.assertIn("语种: rus", res.usage_summary)

            p_402 = ElevenLabsTranscribeProvider(
                settings,
                multipart_fn=mock.Mock(
                    side_effect=ProviderApiError("quota_exceeded", "Not enough credits", http_status=402)
                ),
            )
            with self.assertRaises(ProviderApiError) as err_402:
                p_402.transcribe_chunk(mp3, model="scribe_v3", max_retries=3, sleep_fn=sleeps.append)
            self.assertIn("ElevenLabs Credits 不足", err_402.exception.message)

            p_401 = ElevenLabsTranscribeProvider(
                settings,
                multipart_fn=mock.Mock(
                    side_effect=ProviderApiError("detected_unusual_activity", "Unusual activity detected", http_status=401)
                ),
            )
            with self.assertRaises(ProviderApiError) as err_401:
                p_401.transcribe_chunk(mp3, model="scribe_v3", max_retries=3, sleep_fn=sleeps.append)
            self.assertIn("VPN", err_401.exception.message)

    def test_openai_gpt62_responses_prompt_caching_and_compat_chat(self) -> None:
        captured: list[tuple[str, dict, dict]] = []

        def fake_post(url: str, payload: dict, *, headers: dict | None = None, **_kw) -> dict:
            captured.append((url, payload, headers or {}))
            if url.endswith("/responses"):
                return {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": json.dumps({"keep": [[1, "你好"]], "drop": []})}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 60,
                        "input_tokens_details": {"cached_tokens": 420},
                    },
                    "service_tier": "default",
                }
            return {
                "choices": [{"message": {"content": json.dumps({"keep": [[1, "你好compat"]], "drop": []})}}],
                "usage": {"prompt_tokens": 300, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 100}},
            }

        oa_settings = ProviderSettings(
            name="openai",
            protocol="openai",
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
            api_mode="responses",
        )
        oa = OpenAITranslateProvider(oa_settings, post_fn=fake_post)
        res = oa.generate_json(
            ModelSpec(provider="openai", model="gpt-6.2-sol", service_tier="flex"),
            system_prompt="SYS_PROMPT",
            user_prompt='{"subtitles": [[1, "Hi"]]}',
        )
        self.assertEqual(res.parsed, {"keep": [[1, "你好"]], "drop": []})
        self.assertEqual(res.cached_tokens, 420)
        url0, pay0, hdr0 = captured[0]
        self.assertEqual(url0, "https://api.openai.com/v1/responses")
        self.assertEqual(hdr0["Authorization"], "Bearer sk-test")
        # System prompt must be first developer item in input (for GPT-6.2 Prompt Caching), not top-level instructions
        self.assertNotIn("instructions", pay0)
        self.assertEqual(pay0["input"][0], {"role": "developer", "content": "SYS_PROMPT"})
        self.assertEqual(pay0["reasoning"], {"effort": "low"})
        self.assertNotIn("temperature", pay0)
        self.assertNotIn("service_tier", pay0)  # flex stripped on gpt-6.2-sol

        # gpt-6.2-sol must reject reasoning_effort='none'
        with self.assertRaises(ValueError):
            oa.generate_json(
                ModelSpec(provider="openai", model="gpt-6.2-sol", reasoning_effort="none"),
                system_prompt="SYS",
                user_prompt="{}",
            )

        # OpenAI-compatible provider defaults to /chat/completions and includes temperature for non-reasoning models
        compat_settings = ProviderSettings(
            name="deepseek",
            protocol="openai_compatible",
            api_key="ds-key",
            api_base="https://api.deepseek.com/v1",
        )
        compat = OpenAICompatTranslateProvider(compat_settings, post_fn=fake_post)
        res_c = compat.generate_json(
            ModelSpec(provider="deepseek", model="deepseek-chat"),
            system_prompt="SYS",
            user_prompt="{}",
        )
        self.assertEqual(res_c.parsed["keep"][0][1], "你好compat")
        url1, pay1, _hdr1 = captured[1]
        self.assertEqual(url1, "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(pay1["temperature"], 0.2)
        self.assertNotIn("reasoning_effort", pay1)

    def test_openai_stt_rejects_gpt_transcribe_and_parses_whisper_verbose_json(self) -> None:
        oa_settings = ProviderSettings(
            name="openai",
            protocol="openai",
            api_key="sk-test",
            api_base="https://api.openai.com/v1",
        )
        stt = OpenAICompatTranscribeProvider(
            oa_settings,
            multipart_fn=mock.Mock(
                return_value={
                    "language": "english",
                    "duration": 4.2,
                    "words": [
                        {"word": "Hello", "start": 0.1, "end": 0.4},
                        {"word": "world", "start": 0.5, "end": 0.9},
                    ],
                }
            ),
        )
        self.assertEqual(stt.max_chunk_bytes, 24 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as td:
            mp3 = Path(td) / "a.mp3"
            mp3.write_bytes(b"MP3")
            for bad_model in ("gpt-transcribe", "gpt-live-transcribe", "gpt-4o-transcribe"):
                with self.assertRaises(ValueError):
                    stt.transcribe_chunk(mp3, model=bad_model)

            res = stt.transcribe_chunk(mp3, model="whisper-1")
            self.assertEqual([w.text for w in res.words], ["Hello", "world"])

        # Segment fallback when `words[]` is absent
        seg_words = extract_whisper_verbose_words({"segments": [{"start": 1.0, "end": 2.5, "text": "Fallback line"}]})
        self.assertEqual(len(seg_words), 1)
        self.assertEqual(seg_words[0].text, "Fallback line")

    def test_cross_provider_translation_fallback_openai_to_gemini(self) -> None:
        calls: list[str] = []

        def fake_post(url: str, payload: dict, **_kw) -> dict:
            if "api.openai.com" in url:
                calls.append("openai/gpt-6.2-sol")
                raise ProviderApiError("insufficient_quota", "quota exhausted", http_status=429, error_type="insufficient_quota")
            calls.append("gemini/gemini-4-hydrogen")
            body = json.loads(payload["contents"][0]["parts"][0]["text"])
            kept = [[cid, f"ZH_{cid}"] for cid, _ in body["subtitles"]]
            return _mk_resp({"keep": kept, "drop": []})

        pmap = {
            "openai": ProviderSettings(name="openai", protocol="openai", api_key="sk", api_base="https://api.openai.com/v1", api_mode="responses"),
            "gemini": ProviderSettings(name="gemini", protocol="gemini", api_key="gk", api_base="https://generativelanguage.googleapis.com/v1beta"),
        }
        tr = SubtitleTranslator(
            models=[
                {"provider": "openai", "model": "gpt-6.2-sol", "reasoning_effort": "low"},
                {"provider": "gemini", "model": "gemini-4-hydrogen", "service_tier": "standard"},
            ],
            providers_map=pmap,
            retry_503=2,
            post_fn=fake_post,
            sleep_fn=lambda _: None,
        )
        out = tr.translate_cues([SrtCue(1, 0.0, 1.0, "a"), SrtCue(2, 1.0, 2.0, "b")], system_prompt="sys", max_cues_per_batch=1)
        self.assertEqual(len(out), 2)
        # Batch 1 hits OpenAI -> 429 quota -> switches to Gemini; Batch 2 skips OpenAI directly
        self.assertEqual(calls, ["openai/gpt-6.2-sol", "gemini/gemini-4-hydrogen", "gemini/gemini-4-hydrogen"])

        # Missing OpenAI key in mixed chain skips OpenAI cleanly and keeps Gemini active
        defaults = {
            "providers": {
                "openai": {"api_key": ""},
                "gemini": {"api_key": "gk-only"},
            }
        }
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch("pathlib.Path.home", side_effect=RuntimeError("no home")):
                specs, resolved_map = _resolve_translate_models_and_providers(
                    defaults,
                    {"models": ["openai/gpt-6.2-sol", "gemini/gemini-4-hydrogen"]},
                    {},
                )
        self.assertEqual([s.provider for s in specs], ["gemini"])
        self.assertEqual(list(resolved_map.keys()), ["gemini"])

    def test_multi_provider_cli_parsing_and_providers_command(self) -> None:
        parser = autosub_main.build_parser()

        a_prov = parser.parse_args(["providers"])
        self.assertEqual(a_prov.command, "providers")

        a_tr = parser.parse_args([
            "transcript", "vod.mp4",
            "--provider", "elevenlabs",
            "--model", "scribe_v3",
            "--lang", "rus",
            "--secondary-lang", "eng",
            "--keyterm", "Visage",
            "--keyterm", "Roshan",
            "--diarize",
            "--no-verbatim",
            "--numbers-format", "digits",
        ])
        self.assertEqual(a_tr.provider, "elevenlabs")
        self.assertEqual(a_tr.model, "scribe_v3")
        self.assertEqual(a_tr.lang, "rus")
        self.assertEqual(a_tr.secondary_lang, ["eng"])
        self.assertEqual(a_tr.keyterm, ["Visage", "Roshan"])
        self.assertIs(a_tr.diarize, True)
        self.assertIs(a_tr.no_verbatim, True)
        self.assertEqual(a_tr.numbers_format, "digits")

        a_tl = parser.parse_args([
            "translate", "demo.srt",
            "--provider", "openai",
            "--model", "openai/gpt-6.2-sol,gemini/gemini-4-hydrogen",
            "--reasoning-effort", "medium",
        ])
        self.assertEqual(a_tl.provider, "openai")
        self.assertEqual(a_tl.model, "openai/gpt-6.2-sol,gemini/gemini-4-hydrogen")
        self.assertEqual(a_tl.reasoning_effort, "medium")

        a_run = parser.parse_args([
            "run", "vod.mp4",
            "--transcript-provider", "elevenlabs",
            "--transcript-model", "elevenlabs/scribe_v3",
            "--translate-provider", "openai",
            "--translate-model", "openai/gpt-6.2-sol",
            "--reasoning-effort", "high",
        ])
        self.assertEqual(a_run.transcript_provider, "elevenlabs")
        self.assertEqual(a_run.transcript_model, "elevenlabs/scribe_v3")
        self.assertEqual(a_run.translate_provider, "openai")
        self.assertEqual(a_run.translate_model, "openai/gpt-6.2-sol")
        self.assertEqual(a_run.reasoning_effort, "high")

        # Backward-compatible `--model` alias on `run`
        a_run_legacy = parser.parse_args(["run", "vod.mp4", "--model", "gemini-3.5-transcribe"])
        self.assertEqual(a_run_legacy.transcript_model, "gemini-3.5-transcribe")

        # `providers` CLI exits 0 cleanly
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            rc = autosub_main.main(["providers"])
        self.assertEqual(rc, 0)
        out_txt = buf.getvalue()
        self.assertIn("gemini", out_txt)
        self.assertIn("elevenlabs", out_txt)
        self.assertIn("openai", out_txt)
        self.assertIn("openai_compatible", out_txt)


class TestPhaseOneQualityUpgrades(unittest.TestCase):
    def test_clamp_stretched_words_and_orphan_tail_absorption(self) -> None:
        # 1. Trailing-silence stretch (like Gemini 14.48s single word) clamped anchored at start
        words = [
            {"seg": 0, "start": 10.0, "end": 24.48, "text": "Седьмая.", "space_before": True},
            {"seg": 1, "start": 30.0, "end": 34.36, "text": "Очень", "space_before": True},
            {"seg": 1, "start": 34.40, "end": 34.90, "text": "плохо.", "space_before": True},
        ]
        clamped = clamp_stretched_words(words)
        self.assertAlmostEqual(clamped[0]["start"], 10.0, places=2)
        self.assertAlmostEqual(clamped[0]["end"], 11.86, places=2)  # max(1.8, 0.12*8 + 0.9) = 1.86
        # Anchored to next word -> start trimmed instead of end
        self.assertAlmostEqual(clamped[1]["end"], 34.36, places=2)
        self.assertAlmostEqual(clamped[1]["start"], 32.56, places=2)

        # 2. Comma at >=48 chars followed by 1-word orphan tail (`мне.`) absorbed instead of split
        orphan_words = [
            {"seg": 0, "start": 0.0, "end": 0.4, "text": "Кам,", "space_before": True},
            {"seg": 0, "start": 0.45, "end": 0.8, "text": "кам,", "space_before": True},
            {"seg": 0, "start": 0.85, "end": 1.3, "text": "ребята,", "space_before": True},
            {"seg": 0, "start": 1.35, "end": 1.8, "text": "пурдж", "space_before": True},
            {"seg": 0, "start": 1.85, "end": 2.2, "text": "крипа", "space_before": True},
            {"seg": 0, "start": 2.25, "end": 2.7, "text": "приведите,", "space_before": True},
            {"seg": 0, "start": 2.75, "end": 3.2, "text": "пожалуйста,", "space_before": True},
            {"seg": 0, "start": 3.26, "end": 3.48, "text": "мне.", "space_before": True},
        ]
        cues = group_words_to_cues(orphan_words)
        self.assertEqual(len(cues), 1)
        self.assertTrue(cues[0].text.endswith("пожалуйста, мне."))

    def test_polish_srt_timing_linger_bridge_and_no_overlap(self) -> None:
        raw = [
            (1.00, 1.45, ["为什么要这么打呢？", "Why should we do that?"]),  # 0.45s, 9 zh chars -> needs linger
            (5.00, 5.50, ["快退。", "Back."]),
            (5.60, 6.20, ["小心雾。", "Careful smoke."]),  # 0.10s gap -> bridged seamlessly after linger
        ]
        polished = polish_srt_timing(raw, min_duration=1.0, max_cps=13.0, max_linger=1.2, bridge_gap=0.15, lead_in=0.05)
        self.assertEqual(len(polished), 3)
        # Cue 1 gets lead-in (0.95) and lingers >= 1.0s
        self.assertAlmostEqual(polished[0][0], 0.95, places=3)
        self.assertGreaterEqual(polished[0][1] - polished[0][0], 1.0)
        # Cue 2 cannot overlap Cue 3; micro-gap bridged cleanly
        self.assertLessEqual(polished[1][1], polished[2][0])
        self.assertAlmostEqual(polished[1][1], polished[2][0], places=3)
        # Cue 3 extended into trailing silence
        self.assertGreaterEqual(polished[2][1] - polished[2][0], 1.0)

    def test_align_reference_srt_dual_asr_matching(self) -> None:
        primary = [
            SrtCue(1, 10.0, 11.2, "А, вы снова когда угораете?"),
            SrtCue(2, 20.0, 21.0, "Привет!"),
            SrtCue(3, 30.0, 31.5, "Guys, just chill."),
        ]
        ref = [
            SrtCue(1, 9.9, 11.3, "А, вы с моего КД угораете?"),
            SrtCue(2, 20.0, 21.0, "привет"),
        ]
        aligned = align_reference_srt(primary, ref)
        self.assertEqual(aligned, {1: "А, вы с моего КД угораете?"})

    def test_semantic_cue_merge_and_guardrails(self) -> None:
        batch = [
            SrtCue(1, 1.0, 1.4, "Дабл"),
            SrtCue(2, 2.2, 3.0, "ю, дабл ю, братан!"),
            SrtCue(3, 10.0, 11.0, "Far apart A"),
            SrtCue(4, 16.0, 17.0, "Far apart B"),
            SrtCue(5, 20.0, 20.5, "Опа!"),
        ]
        parsed = {
            "keep": [
                [[1, 2], "W！W！兄弟！", "Дабл ю, дабл ю, братан!"],
                # Illegal merge across 5s silence gap -> must be rejected by guardrail
                [[3, 4], "Should be rejected"],
            ],
            "drop": [5],
        }
        entries, kept, dropped, merged = _parse_cue_entries(
            parsed, batch, allow_merge=True, max_merge_gap=1.8, max_merge_duration=6.5,
        )
        self.assertEqual(merged, 1)
        self.assertEqual(dropped, 1)
        self.assertEqual(kept, 4)
        # Entry 0: merged [1, 2] spans [1.0, 3.0]
        self.assertEqual(entries[0]["ids"], [1, 2])
        self.assertAlmostEqual(entries[0]["start"], 1.0)
        self.assertAlmostEqual(entries[0]["end"], 3.0)
        self.assertEqual(entries[0]["target"], "W！W！兄弟！")
        # Rejected merge [3, 4] safely preserves cues 3 & 4 individually
        self.assertEqual([e["ids"] for e in entries[1:]], [[3], [4]])
        self.assertEqual(entries[1]["target"], "Far apart A")
        self.assertEqual(entries[2]["target"], "Far apart B")

    def test_pass0_auto_context_dual_asr_merge_and_timing_polish_end_to_end(self) -> None:
        calls: list[dict] = []

        def fake_post(url: str, payload: dict, **_kw) -> dict:
            calls.append(payload)
            if len(calls) == 1:
                # Pass 0 response
                ctx_obj = {
                    "summary": "Dota 2 Huskar mid match vs Monkey King.",
                    "speakers_and_names": ["4oapm (streamer)", "Lavrik (teammate)"],
                    "key_entities": ["Huskar", "Monkey King"],
                    "asr_error_map": [["снова когда", "с моего КДА (KDA)"]],
                }
                return {
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(ctx_obj, ensure_ascii=False)}]}],
                    "usage": {"input_tokens": 120, "output_tokens": 50},
                }
            # Pass 1 translation response with merge + ASR correction
            tl_obj = {
                "keep": [
                    [[1, 2], "第一把热热手，不行下一把。", "Ну, первая раскликачная, некст пойдем."],
                    [3, "哦，你们是在笑我的KDA啊？", "А, вы с моего КДА угораете?"],
                ],
                "drop": [],
            }
            return {
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(tl_obj, ensure_ascii=False)}]}],
                "usage": {"input_tokens": 400, "output_tokens": 70, "input_tokens_details": {"cached_tokens": 280}},
            }

        pmap = {
            "openai": ProviderSettings(
                name="openai", protocol="openai", api_key="sk", api_base="https://api.openai.com/v1", api_mode="responses"
            )
        }
        tr = SubtitleTranslator(
            models=[{"provider": "openai", "model": "gpt-6.2-sol", "reasoning_effort": "low"}],
            providers_map=pmap,
            post_fn=fake_post,
            sleep_fn=lambda _: None,
        )
        cues = [
            SrtCue(1, 1.0, 2.0, "Ну, первая раскликачная,"),
            SrtCue(2, 2.06, 2.40, "некст пойдем."),
            SrtCue(3, 10.0, 10.45, "А, вы снова когда угораете?"),
        ]
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "polished.srt"
            out = tr.translate_cues(
                cues,
                system_prompt="BASE_SYS",
                allow_merge=True,
                ref_asr_map={3: "А, вы с моего КД угораете?"},
                polish_timing=True,
                auto_context=True,
                output_path=out_path,
            )
            self.assertEqual(len(calls), 2)
            # Pass 1 system prompt includes Pass 0 context, merge rule, and dual-ASR rule
            p1_dev = calls[1]["input"][0]["content"]
            self.assertIn("Auto-Extracted Full-Video Context (Pass 0)", p1_dev)
            self.assertIn("Lavrik (teammate)", p1_dev)
            self.assertIn("Semantic Fragment Merging", p1_dev)
            self.assertIn("Dual-ASR Reference Disambiguation", p1_dev)
            # Pass 1 user payload includes mergeable_pairs and alt_asr while keeping subtitles 2-tuples
            p1_user = json.loads(calls[1]["input"][1]["content"])
            self.assertEqual(p1_user["mergeable_pairs"], [[1, 2]])
            self.assertEqual(p1_user["alt_asr"], {"3": "А, вы с моего КД угораете?"})
            # Output has 2 merged/polished cues; cue 2 extended from 0.45s to >= 1.0s
            self.assertEqual(len(out), 2)
            self.assertGreaterEqual(out[1][1] - out[1][0], 1.0)
            disk_cues = parse_srt(out_path)
            self.assertEqual(len(disk_cues), 2)
            self.assertIn("KDA", disk_cues[1].text)

    def test_phase1_cli_flags_and_dota_announcer_drop_rule(self) -> None:
        parser = autosub_main.build_parser()
        a_tl = parser.parse_args([
            "translate", "el.srt",
            "--ref-srt", "gem.srt",
            "--auto-context",
            "--no-merge-cues",
            "--no-polish-timing",
        ])
        self.assertEqual(a_tl.ref_srt, "gem.srt")
        self.assertIs(a_tl.auto_context, True)
        self.assertIs(a_tl.merge_cues, False)
        self.assertIs(a_tl.polish_timing, False)

        a_run = parser.parse_args(["run", "vod.mp4", "--auto-context", "--no-merge-cues", "--no-polish-timing"])
        self.assertIs(a_run.auto_context, True)
        self.assertIs(a_run.merge_cues, False)
        self.assertIs(a_run.polish_timing, False)

        dota = load_profile("twitch_dota2")
        self.assertIn("in-game announcer", dota.prompt_template)
        self.assertIn("Prepare for battle.", dota.prompt_template)

        g = load_glossary(PROJECT_ROOT / "glossaries" / "dota2_ru_en_zh.yaml")
        self.assertIn("英雄栏/英雄面板", g["mechanics_and_slang"])
        self.assertIn("KDA", g["mechanics_and_slang"])
        self.assertIn("后期能力/成长性", g["mechanics_and_slang"])
        self.assertIn("屏蔽", g["mechanics_and_slang"])
        self.assertIn("开小号/炸鱼", g["mechanics_and_slang"])

    def test_align_reference_srt_denoises_spelling_while_keeping_real_fixes(self) -> None:
        primary = [
            SrtCue(1, 1.0, 2.0, "Да ты пиздишь, что у него такой винрейт хуевый."),
            SrtCue(2, 5.0, 6.0, "Баттерфляй."),
            SrtCue(3, 10.0, 11.0, "Блять, трипа не добил."),
            SrtCue(4, 15.0, 16.5, "Лаврик ебаный рекорд получает."),
            SrtCue(5, 20.0, 22.0, "Что такое надо за, блядь, бегать, я не понимаю."),
        ]
        ref = [
            SrtCue(1, 1.0, 2.0, "Да ты пиздишь, что у него такой винрейт хуёвый."),
            SrtCue(2, 5.0, 6.0, "Батерфляй."),
            SrtCue(3, 10.0, 11.0, "Блядь, крипа не добил."),
            SrtCue(4, 15.0, 16.5, "Лаврик ебаный репорт получает."),
            SrtCue(5, 20.0, 22.0, "Что такое на дазле, блядь, бегать? Я не понимаю."),
        ]
        aligned = align_reference_srt(primary, ref)
        # Pure spelling/ё/doubled-letter diffs dropped
        self.assertNotIn(1, aligned)
        self.assertNotIn(2, aligned)
        # Genuine lexical fixes preserved 100%
        self.assertIn(3, aligned)
        self.assertIn(4, aligned)
        self.assertIn(5, aligned)

    def test_yaml_dual_asr_resolution_pipeline_and_cli_flags(self) -> None:
        providers_cfg = get_providers_config({})
        primary = ModelSpec(provider="elevenlabs", model="scribe_v3")

        # 1. Disabled by default
        self.assertIsNone(resolve_dual_asr_spec({"ref_enabled": False}, providers_cfg, primary_spec=primary))

        # 2. Enabled in YAML with ref_model -> auto-infers gemini provider
        spec = resolve_dual_asr_spec(
            {"ref_enabled": True, "ref_model": "gemini-3.5-transcribe"},
            providers_cfg,
            primary_spec=primary,
        )
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual((spec.provider, spec.model), ("gemini", "gemini-3.5-transcribe"))

        # 3. CLI --no-ref-transcript disables even when YAML ref_enabled=True
        self.assertIsNone(
            resolve_dual_asr_spec(
                {"ref_enabled": True, "ref_model": "gemini-3.5-transcribe"},
                providers_cfg,
                primary_spec=primary,
                ref_enabled_override=False,
            )
        )

        # 4. Duplicate primary == ref skipped cleanly
        self.assertIsNone(
            resolve_dual_asr_spec(
                {"ref_enabled": True, "ref_model": "elevenlabs/scribe_v3"},
                providers_cfg,
                primary_spec=primary,
            )
        )

        # 5. ref_enabled=True without ref_model raises clear ValueError
        with self.assertRaises(ValueError):
            resolve_dual_asr_spec({"ref_enabled": True, "ref_model": ""}, providers_cfg, primary_spec=primary)

        # 6. CLI flags parsing
        parser = autosub_main.build_parser()
        a_tr = parser.parse_args([
            "transcript", "vod.mp4", "--model", "elevenlabs/scribe_v3",
            "--ref-model", "gemini-3.5-transcribe", "--ref-output", "ref.srt",
        ])
        self.assertEqual(a_tr.ref_model, "gemini-3.5-transcribe")
        self.assertEqual(a_tr.ref_output, "ref.srt")

        a_run = parser.parse_args([
            "run", "vod.mp4", "--ref-transcript-model", "gemini-3.5-transcribe",
            "--ref-srt-output", "mid_ref.srt", "--no-ref-transcript",
        ])
        self.assertEqual(a_run.ref_model, "gemini-3.5-transcribe")
        self.assertEqual(a_run.ref_srt_output, "mid_ref.srt")
        self.assertIs(a_run.ref_enabled, False)

        # 7. End-to-end offline single-VAD dual-ASR pipeline via YAML config
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            cfg_path = d / "defaults.yaml"
            cfg_path.write_text(
                yaml.safe_dump({
                    "providers": {
                        "elevenlabs": {"api_key": "el-k"},
                        "gemini": {"api_key": "gm-k"},
                    },
                    "transcript": {
                        "vad_mode": "none",
                        "provider": "elevenlabs",
                        "model": "scribe_v3",
                        "ref_enabled": True,
                        "ref_model": "gemini-3.5-transcribe",
                    },
                }),
                encoding="utf-8",
            )
            media = d / "clip.wav"
            media.write_bytes(b"RIFF")

            def fake_subproc(cmd, **_kw):
                out_f = Path(cmd[-1])
                # 1s of 16kHz float32 zeros
                out_f.write_bytes(b"\x00" * (16000 * 4))

            el_resp = {"language_code": "rus", "words": [{"text": "на", "start": 0.1, "end": 0.3, "type": "word"}, {"text": " ", "start": 0.3, "end": 0.35, "type": "spacing"}, {"text": "сцене", "start": 0.35, "end": 0.8, "type": "word"}]}
            gm_resp = {"steps": [{"content": [{"annotations": [{"type": "word_info", "text": "на", "start_offset": "0.1s", "end_offset": "0.3s"}, {"type": "word_info", "text": "Свене", "start_offset": "0.35s", "end_offset": "0.8s"}]}]}]}

            with (
                mock.patch("src.transcribe.subprocess.run", side_effect=fake_subproc),
                mock.patch("src.transcribe.render_chunk_mp3", side_effect=lambda pcm, cm, out_p, **kw: out_p.write_bytes(b"MP3")),
                mock.patch("src.providers.elevenlabs.post_multipart", return_value=el_resp),
                mock.patch("src.providers.gemini.upload_resumable_file", return_value={"name": "files/1", "uri": "https://x/1"}),
                mock.patch("src.providers.gemini.call_gemini_interactions_transcribe", return_value=gm_resp),
                mock.patch("src.providers.gemini.delete_url"),
            ):
                p_srt, r_srt = transcribe_media_pair(media, defaults_path=cfg_path, output_path=d / "out.srt")
            self.assertIsNotNone(p_srt)
            self.assertIsNotNone(r_srt)
            assert p_srt is not None and r_srt is not None
            self.assertTrue(p_srt.is_file() and r_srt.is_file())
            self.assertEqual(r_srt.name, "out_ref.srt")
            self.assertIn("на сцене", p_srt.read_text(encoding="utf-8"))
            self.assertIn("на Свене", r_srt.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
