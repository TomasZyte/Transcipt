import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import asr_comparison as adapter


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root / "context.wav"
        self.make_audio(1)
        self.manifest = self.root / "manifest.json"
        self.data = {"clips": [{"id": "context", "file": "context.wav", "sha256": adapter.digest(self.audio),
                     "duration_seconds": 1, "source_start_seconds": 8.5, "source_end_seconds": 9.5}]}
        self.save()

    def make_audio(self, seconds, channels=1):
        with wave.open(str(self.audio), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * (int(seconds * 16000) * channels))

    def save(self):
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")

    def plan(self, variant="fw_v3"):
        return adapter.prepare(self.manifest, "context", variant)

    def asset_lock(self):
        bundle = self.root / "model"
        bundle.mkdir()
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (bundle / name).write_bytes(b"offline-test-fixture")
        lock = {"model": "large-v3", "revision": "a" * 40, "root": "model",
                "packages": {"faster-whisper": "1.2.1", "ctranslate2": "4.5.0", "torch": "2.8.0"},
                "files": {p.name: adapter.digest(p) for p in bundle.iterdir()}}
        path = self.root / "lock.json"
        path.write_text(json.dumps(lock), encoding="utf-8")
        return path, lock

    def test_default_preparation_imports_no_models_and_keeps_unknown_results(self):
        with patch.object(adapter.importlib, "import_module", side_effect=AssertionError("No ML imports")):
            for variant in adapter.VARIANTS:
                result = self.plan(variant)
                self.assertEqual(result["status"], "not_run")
                self.assertIsNone(result["raw_asr_text"])
                self.assertIsNone(result["wer"])
                self.assertIsNone(result["billed_cost_usd"])

    def test_checksum_rejected(self):
        self.data["clips"][0]["sha256"] = "0" * 64
        self.save()
        with self.assertRaises(ValueError):
            self.plan()

    def test_invalid_paths_and_duplicate_clip_rejected(self):
        for value in ("../other.wav", "C:/sound.wav", "..\\sound.wav"):
            self.data["clips"][0]["file"] = value
            self.save()
            with self.assertRaises(ValueError):
                self.plan()
        self.data["clips"] *= 2
        self.save()
        with self.assertRaises(ValueError):
            self.plan()

    def test_format_duration_offsets_and_truncation(self):
        for value in (True, float("nan"), -1, 12):
            self.data["clips"][0]["source_start_seconds"] = value
            self.save()
            with self.assertRaises(ValueError):
                self.plan()
        self.data["clips"][0]["source_start_seconds"] = 8.5
        self.make_audio(1, channels=2)
        self.data["clips"][0]["sha256"] = adapter.digest(self.audio)
        self.save()
        with self.assertRaises(ValueError):
            self.plan()
        self.make_audio(1)
        self.audio.write_bytes(self.audio.read_bytes()[:-2])
        self.data["clips"][0]["sha256"] = adapter.digest(self.audio)
        self.save()
        with self.assertRaises(ValueError):
            self.plan()

    def test_gigaam_full_recording_limit_before_import(self):
        self.make_audio(26)
        self.data["clips"][0].update(sha256=adapter.digest(self.audio), duration_seconds=26, source_end_seconds=34.5)
        self.save()
        with self.assertRaises(ValueError):
            self.plan("gigaam_v3")
        self.assertEqual(self.plan("fw_v3")["duration_seconds"], 26)

    def test_configs_are_isolated_and_no_reference_prompt_is_used(self):
        first = adapter.config_for("wx_ru_vad")
        first["load"]["asr_options"]["initial_prompt"] = "reference text"
        self.assertIsNone(adapter.config_for("wx_ru_vad")["load"]["asr_options"]["initial_prompt"])
        self.assertEqual(adapter.config_for("wx_auto")["load"]["vad_options"]["vad_onset"], .5)
        self.assertEqual(adapter.config_for("wx_ru_vad")["load"]["vad_options"]["vad_onset"], .3)
        self.assertFalse(adapter.config_for("fw_v2")["transcribe"]["vad_filter"])
        with self.assertRaises(ValueError):
            self.plan("unapproved-model")

    def test_reference_fields_never_reach_model_config(self):
        self.data["clips"][0]["initial_prompt"] = "secret reference phrase"
        self.data["owner_reference"] = "secret reference phrase"
        self.save()
        self.assertNotIn("secret reference phrase", json.dumps(self.plan()))

    def test_execution_guard_precedes_assets_and_model_import(self):
        with patch.dict(adapter.os.environ, {}, clear=True), patch.object(adapter, "verify_assets") as verify:
            with self.assertRaises(ValueError):
                adapter.run(self.plan(), "missing", self.root / "result.json")
            verify.assert_not_called()

    def test_execution_rejects_tampered_config(self):
        plan = self.plan()
        plan["config"]["transcribe"]["initial_prompt"] = "reference"
        with patch.dict(adapter.os.environ, {"ASR_COMPARISON_TEST": "1"}), patch.object(adapter, "verify_assets") as verify:
            with self.assertRaises(ValueError):
                adapter.run(plan, "missing", self.root / "result.json")
            verify.assert_not_called()

    def test_bundle_hashes_and_package_versions(self):
        path, lock = self.asset_lock()
        with patch.object(adapter.metadata, "version", side_effect=lock["packages"].get):
            verified = adapter.verify_assets(path, self.plan()["config"])
            self.assertEqual(verified["revision"], "a" * 40)
            (self.root / "model/model.bin").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                adapter.verify_assets(path, self.plan()["config"])

    def test_extra_bundle_files_or_runtime_drift_rejected(self):
        path, lock = self.asset_lock()
        with patch.object(adapter.metadata, "version", return_value="wrong"):
            with self.assertRaises(ValueError):
                adapter.verify_assets(path, self.plan()["config"])
        (self.root / "model/extra.json").write_text("{}")
        with self.assertRaises(ValueError):
            adapter.verify_assets(path, self.plan()["config"])

    def test_distribution_names_are_canonicalized(self):
        path, lock = self.asset_lock()
        lock["packages"]["Faster_Whisper"] = lock["packages"].pop("faster-whisper")
        path.write_text(json.dumps(lock), encoding="utf-8")
        with patch.object(adapter.metadata, "version", side_effect=lock["packages"].get):
            self.assertEqual(adapter.verify_assets(path, self.plan()["config"])["revision"], "a" * 40)

    def test_offline_guard_blocks_connections_and_restores_environment(self):
        before = dict(adapter.os.environ)
        with adapter.offline_models():
            self.assertEqual(adapter.os.environ["HF_HUB_OFFLINE"], "1")
            with self.assertRaises(RuntimeError):
                socket.create_connection(("example.com", 443))
        self.assertEqual(dict(adapter.os.environ), before)

    def test_faster_whisper_consumes_generator_and_preserves_settings(self):
        seen = []
        def segments():
            seen.append("consumed")
            yield SimpleNamespace(start=.1, end=.8, text=" test", avg_logprob=-.2, no_speech_prob=.01)
        model = Mock()
        model.transcribe.return_value = segments(), SimpleNamespace(language="ru")
        engine = SimpleNamespace(WhisperModel=Mock(return_value=model))
        with patch.object(adapter.importlib, "import_module", return_value=engine):
            result = adapter.backend(adapter.config_for("fw_v3"), self.audio, {"root": "local-model"})
        self.assertEqual(seen, ["consumed"])
        self.assertEqual(result["text"], "test")
        self.assertFalse(model.transcribe.call_args.kwargs["vad_filter"])
        self.assertEqual(model.transcribe.call_args.kwargs["temperature"], 0)
        self.assertTrue(engine.WhisperModel.call_args.kwargs["local_files_only"])

    def test_whisperx_uses_local_vad_and_records_raw_text(self):
        model = Mock(options=SimpleNamespace())
        model.transcribe.return_value = {"segments": [{"start": .1, "end": .8, "text": "test"}], "language": "ru"}
        engine = SimpleNamespace(load_model=Mock(return_value=model))
        vad = SimpleNamespace(load_vad_model=Mock(return_value="vad-instance"))
        with patch.object(adapter.importlib, "import_module", side_effect=[engine, vad]):
            result = adapter.backend(adapter.config_for("wx_ru_vad"), self.audio, {"root": "model", "vad_path": "vad.bin"})
        self.assertEqual(result["text"], "test")
        self.assertEqual(vad.load_vad_model.call_args.kwargs["vad_onset"], .3)
        self.assertEqual(engine.load_model.call_args.kwargs["vad_model"], "vad-instance")
        self.assertEqual(model.transcribe.call_args.kwargs["language"], "ru")

    def test_gigaam_uses_named_cached_checkpoint_and_unavailable_timing(self):
        model = Mock()
        model.parameters.return_value = [SimpleNamespace(dtype="float16")]
        model.transcribe.return_value = SimpleNamespace(text="test")
        engine = SimpleNamespace(load_model=Mock(return_value=model))
        with patch.object(adapter.importlib, "import_module", return_value=engine):
            raw = adapter.backend(adapter.config_for("gigaam_v3"), self.audio, {"root": "model-cache"})
        self.assertEqual(engine.load_model.call_args.args[0], "v3_rnnt")
        result = adapter.normalize_result(self.plan("gigaam_v3"), raw)
        self.assertIsNone(result["raw_asr_segments"][0]["source_start"])
        self.assertIsNone(result["language_detected"])

    def test_offsets_preserve_source_time_and_reject_invalid_timestamps(self):
        raw = {"text": "test", "segments": [{"start": .1, "end": .8, "text": "test"}]}
        result = adapter.normalize_result(self.plan(), raw)
        self.assertAlmostEqual(result["raw_asr_segments"][0]["source_start"], 8.6)
        self.assertIsNone(result["wer"])
        raw["segments"][0]["end"] = float("nan")
        with self.assertRaises(ValueError):
            adapter.normalize_result(self.plan(), raw)

    def test_failed_attempt_cannot_implicitly_retry_or_leak_error(self):
        output = self.root / "result.json"
        assets = {"lock_sha256": "a" * 64}
        with patch.dict(adapter.os.environ, {"ASR_COMPARISON_TEST": "1"}), \
             patch.object(adapter, "verify_assets", return_value=assets), \
             patch.object(adapter, "backend", side_effect=RuntimeError("secret-url")) as backend:
            with self.assertRaises(RuntimeError):
                adapter.run(self.plan(), "unused", output)
            with self.assertRaises(ValueError):
                adapter.run(self.plan(), "unused", output)
            self.assertEqual(backend.call_count, 1)
        self.assertNotIn("secret-url", Path(str(output) + ".failed.json").read_text())

    def test_existing_results_are_never_overwritten(self):
        output = self.root / "result.json"
        adapter.write_new(output, {"original": True})
        with self.assertRaises(FileExistsError):
            adapter.write_new(output, {"original": False})
        self.assertEqual(adapter.read_json(output), {"original": True})


if __name__ == "__main__":
    unittest.main()
