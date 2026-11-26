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
from src.config import PROJECT_ROOT, load_engine_config, resolve_api_key, resolve_binary
from src.http_client import GeminiApiError, post_json, upload_resumable_file
from src.profile import (
    list_profiles,
    load_glossary,
    load_profile,
    merge_glossaries,
    parse_var_overrides,
    render_glossary_section,
    render_prompt,
)
from src.srt import (
    SrtCue,
    append_srt_cues,
    format_srt_ts,
    group_words_to_cues,
    map_word_to_original,
    parse_srt,
    parse_srt_ts,
    write_srt,
)
from src.transcribe import call_gemini_transcribe, extract_words_from_interaction
from src.translate import (
    DEFAULT_TRANSLATE_MODELS,
    SubtitleTranslator,
    _parse_compact_translation,
    format_cue_lines,
    normalize_translate_chain,
    split_cues_evenly,
)
from src.vad import find_silero_onnx_model, pack_segments_into_chunks


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
            fake_ffmpeg = app_root / "bin" / "ffmpeg.exe"
            fake_ffmpeg.write_bytes(b"MZ")
            fake_onnx = bundle_root / "models" / "silero_vad.onnx"
            fake_onnx.write_bytes(b"ONNX")

            resolved = resolve_binary(
                "",
                ["ffmpeg.exe", "ffmpeg"],
                app_root=app_root,
                bundle_root=bundle_root,
            )
            self.assertEqual(Path(resolved), fake_ffmpeg)

            with mock.patch.dict("os.environ", {}, clear=True):
                found_onnx = find_silero_onnx_model(app_root=app_root, bundle_root=bundle_root)
            self.assertEqual(found_onnx, fake_onnx)

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


if __name__ == "__main__":
    unittest.main()
