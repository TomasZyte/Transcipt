"""Test-only batch entry point. It is not copied into the production image."""
import hashlib
from importlib.metadata import version
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import unquote, urlsplit
import urllib.request
import wave

MAX_BYTES = 2 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_request(inp, storage_host):
    if not re.fullmatch(r"[a-f0-9]{32}\.r2\.cloudflarestorage\.com", storage_host):
        raise ValueError("Set the explicit test R2 host")
    if not isinstance(inp, dict) or inp.get("profile") not in ("baseline", "community"):
        raise ValueError("Choose an explicit benchmark profile")
    samples = inp.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 5:
        raise ValueError("Expected 1–5 test samples")
    ids = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Invalid sample")
        sample_id = sample.get("id", "")
        if not isinstance(sample_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", sample_id) or sample_id in ids:
            raise ValueError("Invalid or duplicate sample ID")
        ids.add(sample_id)
        if not isinstance(sample.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", sample["sha256"]):
            raise ValueError("Missing sample checksum")
        url = urlsplit(sample.get("media_url", ""))
        path = unquote(url.path)
        valid_path = (url.netloc == storage_host and path.startswith("/poslovnica-test/quality/")) or (
            url.netloc == "poslovnica-test." + storage_host and path.startswith("/quality/"))
        if url.scheme != "https" or not valid_path or not url.query or url.fragment or ".." in path.split("/"):
            raise ValueError("URL outside the dedicated test bucket/prefix")
    return samples


def fetch_sample(sample, path):
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(sample["media_url"], timeout=30) as response:
        data = response.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != sample["sha256"]:
        raise ValueError("Sample size or checksum mismatch")
    path.write_bytes(data)
    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("Expected mono 16 kHz PCM16 WAV")
        duration = wav.getnframes() / wav.getframerate()
        if not 0 < duration <= 60:
            raise ValueError("Sample duration must be within 60 seconds")
    return duration


def handler(job):
    try:
        if os.getenv("QUALITY_BENCHMARK") != "1":
            raise ValueError("Test benchmark mode is required")
        samples = validate_request(job.get("input"), os.getenv("QUALITY_R2_HOST", ""))
        if job["input"]["profile"] != os.getenv("QUALITY_PROFILE"):
            raise ValueError("Endpoint image does not match the requested benchmark profile")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="quality-") as directory:
            downloaded, total = [], 0
            # Verify the entire set before loading any models.
            for sample in samples:
                path = Path(directory) / (sample["id"] + ".wav")
                duration = fetch_sample(sample, path)
                total += duration
                if total > 180:
                    raise ValueError("Sample set exceeds 180 seconds")
                downloaded.append((sample, path, duration))
            from pipeline import process
            results = []
            for sample, path, duration in downloaded:
                output = process(str(path), {"language": "auto", "enableDiarization": True},
                                 file_name=path.name, file_type="audio")
                results.append({"id": sample["id"], "sha256": sample["sha256"],
                                "durationSeconds": round(duration, 3), "output": output})
            return {"profile": job["input"]["profile"], "samples": results,
                    "elapsedSeconds": round(time.perf_counter() - started, 3),
                    "versions": {p: version(p) for p in ("whisperx", "pyannote.audio", "torch", "transformers")}}
    except Exception as exc:
        # Presigned URLs can occur in transport exceptions. Never return/log them.
        return {"error": "Quality benchmark failed: " + type(exc).__name__}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
