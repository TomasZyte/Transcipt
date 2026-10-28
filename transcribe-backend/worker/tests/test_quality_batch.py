from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_batch_handler as batch

HOST = "0" * 32 + ".r2.cloudflarestorage.com"


def payload():
    return {"profile": "baseline", "samples": [{"id": "sound1", "sha256": "a" * 64,
            "media_url": "https://" + HOST + "/poslovnica-test/quality/abc/sound1.wav?signature=offline"}]}


class BatchValidation(unittest.TestCase):
    def test_accept_only_explicit_test_bucket(self):
        self.assertEqual(len(batch.validate_request(payload(), HOST)), 1)
        for url in ("https://" + HOST + "/transcribe/quality/a.wav?key=x",
                    "http://" + HOST + "/poslovnica-test/quality/a.wav?key=x",
                    "https://example.com/poslovnica-test/quality/a.wav?key=x",
                    "https://" + HOST + "/poslovnica-test/quality/%2e%2e/a.wav?key=x"):
            p = payload()
            p["samples"][0]["media_url"] = url
            with self.assertRaises(ValueError):
                batch.validate_request(p, HOST)

    def test_reject_duplicate_and_traversal_ids(self):
        p = payload()
        p["samples"].append(deepcopy(p["samples"][0]))
        with self.assertRaises(ValueError):
            batch.validate_request(p, HOST)
        p = payload()
        p["samples"][0]["id"] = "../elsewhere"
        with self.assertRaises(ValueError):
            batch.validate_request(p, HOST)

    def test_limit_batch_before_download(self):
        for count in (0, 6):
            p = payload()
            p["samples"] = p["samples"] * count
            with self.assertRaises(ValueError):
                batch.validate_request(p, HOST)

    def test_wrong_image_profile_stops_before_network_or_models(self):
        with patch.dict(batch.os.environ, {"QUALITY_BENCHMARK": "1", "QUALITY_PROFILE": "community", "QUALITY_R2_HOST": HOST}), \
                patch.object(batch, "fetch_sample", side_effect=AssertionError("No network")):
            self.assertIn("error", batch.handler({"input": payload()}))

    def test_requires_explicit_benchmark_mode(self):
        with patch.dict(batch.os.environ, {}, clear=True), \
                patch.object(batch, "fetch_sample", side_effect=AssertionError("No network")):
            self.assertIn("error", batch.handler({"input": payload()}))

    def test_rejects_checksum_mismatch(self):
        response = io.BytesIO(b"wrong content")
        with patch.object(batch.urllib.request, "build_opener") as opener:
            opener.return_value.open.return_value = response
            with self.assertRaises(ValueError):
                batch.fetch_sample(payload()["samples"][0], Path("must-not-be-written.wav"))
        self.assertFalse(Path("must-not-be-written.wav").exists())

    def test_valid_wav_duration_and_hash(self):
        content = io.BytesIO()
        with wave.open(content, "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0" * 32000)
        data = content.getvalue()
        sample = payload()["samples"][0]
        sample["sha256"] = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as folder, patch.object(batch.urllib.request, "build_opener") as opener:
            opener.return_value.open.return_value = io.BytesIO(data)
            self.assertEqual(batch.fetch_sample(sample, Path(folder) / "sample.wav"), 1)


if __name__ == "__main__":
    unittest.main()
