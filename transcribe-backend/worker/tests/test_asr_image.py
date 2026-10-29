from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import asr_comparison as adapter
import asr_comparison_handler as handler
import prepare_asr_assets as assets

HOST = "0" * 32 + ".r2.cloudflarestorage.com"
ENV = {"ASR_COMPARISON_TEST": "1", "ASR_COMPARISON_IMAGE": "1", "ASR_IMAGE_MODEL": "large-v3", "QUALITY_R2_HOST": HOST}


def fixture_wav():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 16000)
    return stream.getvalue()


def request():
    return {"input": {"profile": "asr-comparison", "variant": "wx_auto", "sample": {
        "id": "context", "sha256": hashlib.sha256(fixture_wav()).hexdigest(),
        "media_url": "https://" + HOST + "/poslovnica-test/quality/a.wav?signature=private",
        "source_start_seconds": 8.5, "source_end_seconds": 9.5, "duration_seconds": 1}}}


class HandlerTests(unittest.TestCase):
    def test_disabled_image_stops_before_io(self):
        with patch.dict(handler.os.environ, {}, clear=True), patch.object(handler, "fetch_sample") as fetch:
            self.assertIn("error", handler.handler(request()))
            fetch.assert_not_called()

    def test_wrong_model_stops_before_io(self):
        job = request()
        job["input"]["variant"] = "gigaam_v3"
        with patch.dict(handler.os.environ, ENV), patch.object(handler, "fetch_sample") as fetch:
            self.assertIn("error", handler.handler(job))
            fetch.assert_not_called()

    def test_reject_arbitrary_params_prompt_and_batch(self):
        for field in ("samples", "initial_prompt", "config", "model_lock"):
            job = request()
            job["input"][field] = "untrusted override"
            with patch.dict(handler.os.environ, ENV), patch.object(handler, "fetch_sample") as fetch:
                self.assertIn("error", handler.handler(job))
                fetch.assert_not_called()
        job = request()
        job["input"]["sample"]["reference_text"] = "expected words"
        with patch.dict(handler.os.environ, ENV):
            with self.assertRaises(ValueError):
                handler.validate(job["input"])

    def test_reject_production_r2_and_wrong_duration(self):
        for url in ("https://example.com/a.wav?x=1", "https://" + HOST + "/transcribe/quality/a.wav?x=1"):
            job = request()
            job["input"]["sample"]["media_url"] = url
            with patch.dict(handler.os.environ, ENV):
                with self.assertRaises(ValueError):
                    handler.validate(job["input"])
        for duration in (True, float("nan"), -1, 61, 2):
            job = request()
            job["input"]["sample"]["duration_seconds"] = duration
            with patch.dict(handler.os.environ, ENV):
                with self.assertRaises(ValueError):
                    handler.validate(job["input"])

    def test_fetch_and_run_one_child_without_forwarding_url(self):
        def fetch(sample, path):
            path.write_bytes(fixture_wav())
        def child(command, **kwargs):
            self.assertEqual(kwargs["timeout"], 100)
            self.assertEqual(command.count("--execute"), 1)
            self.assertNotIn("signature=private", " ".join(command))
            manifest = adapter.read_json(command[command.index("--manifest") + 1])
            self.assertNotIn("media_url", manifest["clips"][0])
            output = command[command.index("--output") + 1]
            adapter.write_new(output, {"status": "completed", "variant_id": "wx_auto",
                "audio_sha256": request()["input"]["sample"]["sha256"], "raw_asr_text": "test", "audio_file": "private-temp"})
            return SimpleNamespace(returncode=0)
        with patch.dict(handler.os.environ, ENV), patch.object(handler, "fetch_sample", side_effect=fetch), \
             patch.object(handler.subprocess, "run", side_effect=child) as run:
            result = handler.handler(request())
        self.assertNotIn("error", result)
        self.assertEqual(run.call_count, 1)
        self.assertNotIn("audio_file", result["output"])

    def test_bad_download_prevents_child(self):
        with patch.dict(handler.os.environ, ENV), \
             patch.object(handler, "fetch_sample", side_effect=lambda s, p: p.write_bytes(b"bad")), \
             patch.object(handler.subprocess, "run") as run:
            self.assertIn("error", handler.handler(request()))
            run.assert_not_called()

    def test_timeout_no_retry_and_no_private_error(self):
        with patch.dict(handler.os.environ, ENV), \
             patch.object(handler, "fetch_sample", side_effect=lambda s, p: p.write_bytes(fixture_wav())), \
             patch.object(handler.subprocess, "run", side_effect=subprocess.TimeoutExpired("private-secret", 100)) as run:
            result = handler.handler(request())
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result, {"error": "ASR comparison failed: TimeoutExpired"})

    def test_failed_child_is_not_completed(self):
        with patch.dict(handler.os.environ, ENV), \
             patch.object(handler, "fetch_sample", side_effect=lambda s, p: p.write_bytes(fixture_wav())), \
             patch.object(handler.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
            self.assertEqual(handler.handler(request()), {"error": "ASR comparison failed: RuntimeError"})


class AssetTests(unittest.TestCase):
    def test_default_only_prints_recipe(self):
        with patch.object(sys, "argv", ["prepare_asr_assets.py", "--model", "large-v3"]), \
             patch.object(assets, "assemble") as assemble, redirect_stdout(io.StringIO()) as output:
            assets.main()
            assemble.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["status"], "download_plan_only")

    def test_recipe_has_immutable_urls_and_separate_weights(self):
        a, b = assets.recipe("large-v3"), assets.recipe("large-v2")
        self.assertNotEqual(a["revision"], b["revision"])
        for recipe in (a, b):
            self.assertTrue(all(recipe["revision"] in r["url"] for r in recipe["files"]))
            self.assertTrue(all("/main/" not in r["url"] for r in recipe["files"]))
        self.assertEqual(assets.recipe("v3_rnnt")["files"][0]["name"], "v3_rnnt.ckpt")

    def test_fetch_verifies_exact_size_sha_and_git_blob(self):
        data = b"verified fixture"
        row = {"url": "https://example.com/pinned", "max_bytes": len(data), "size": len(data),
               "sha256": hashlib.sha256(data).hexdigest(),
               "git_blob_sha1": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}
        with tempfile.TemporaryDirectory() as folder, patch.object(assets.urllib.request, "urlopen", return_value=io.BytesIO(data)):
            path = Path(folder) / "file"
            assets.fetch(row, path)
            self.assertEqual(path.read_bytes(), data)

    def test_bad_size_hash_or_upstream_md5_fails(self):
        base = {"url": "https://example.com/pinned", "max_bytes": 20, "size": 4}
        for changed in ({"max_bytes": 2}, {"size": 5}, {"sha256": "0" * 64}, {"git_blob_sha1": "0" * 40}, {"md5": "0" * 32}):
            row = dict(base, **changed)
            with tempfile.TemporaryDirectory() as folder, patch.object(assets.urllib.request, "urlopen", return_value=io.BytesIO(b"test")):
                with self.assertRaises(ValueError):
                    assets.fetch(row, Path(folder) / "file")

    def test_assembly_refuses_existing_destination_before_download(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(assets, "fetch") as fetch:
            with self.assertRaises(FileExistsError):
                assets.assemble("large-v3", Path(folder))
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
