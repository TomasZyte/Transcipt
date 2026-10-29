"""Separate test-only RunPod entry point: one sample/variant per job."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from asr_comparison import config_for, finite, prepare, read_json, write_new
from quality_batch_handler import fetch_sample, validate_request as validate_r2

CHILD_TIMEOUT_SECONDS = 100
SAMPLE_KEYS = {"id", "sha256", "media_url", "source_start_seconds", "source_end_seconds", "duration_seconds"}


def validate(inp):
    if os.getenv("ASR_COMPARISON_TEST") != "1" or os.getenv("ASR_COMPARISON_IMAGE") != "1":
        raise ValueError("Explicit separate test image and execution flag required")
    if not isinstance(inp, dict) or set(inp) != {"profile", "variant", "sample"} or inp["profile"] != "asr-comparison":
        raise ValueError("Expected a single explicit ASR comparison input")
    config = config_for(inp["variant"])
    if config["model"] != os.getenv("ASR_IMAGE_MODEL"):
        raise ValueError("Variant does not match image model")
    sample = inp["sample"]
    if not isinstance(sample, dict) or set(sample) != SAMPLE_KEYS:
        raise ValueError("Unknown or missing sample fields")
    start, end, duration = (sample[k] for k in ("source_start_seconds", "source_end_seconds", "duration_seconds"))
    if not all(finite(n) for n in (start, end, duration)) or not 0 <= start < end or not 0 < duration <= config["max_seconds"] or abs(end - start - duration) > 1 / 16000:
        raise ValueError("Invalid sample duration or source offsets")
    validate_r2({"profile": "baseline", "samples": [sample]}, os.getenv("QUALITY_R2_HOST", ""))
    return sample


def handler(job):
    try:
        sample = validate(job.get("input"))
        variant = job["input"]["variant"]
        with tempfile.TemporaryDirectory(prefix="asr-comparison-") as directory:
            root = Path(directory)
            audio = root / "sample.wav"
            fetch_sample(sample, audio)
            clip = {k: v for k, v in sample.items() if k != "media_url"}
            clip["file"] = audio.name
            manifest = root / "manifest.json"
            output = root / "result.json"
            write_new(manifest, {"clips": [clip]})
            prepare(manifest, clip["id"], variant)  # Reject bad audio before spawning an ML process.
            # Child exits per job: CUDA state and socket guard never affect the SDK process.
            completed = subprocess.run([sys.executable, str(Path(__file__).with_name("asr_comparison.py")),
                "--manifest", str(manifest), "--clip", clip["id"], "--variant", variant,
                "--output", str(output), "--execute", "--model-lock", "/opt/asr/model-lock.json"],
                timeout=CHILD_TIMEOUT_SECONDS, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if completed.returncode != 0:
                raise RuntimeError("ASR child failed")
            result = read_json(output)
            if result.get("status") != "completed" or result.get("variant_id") != variant or result.get("audio_sha256") != clip["sha256"]:
                raise ValueError("Unexpected child result")
            result.pop("audio_file", None)
            return {"profile": "asr-comparison", "sample_id": clip["id"], "output": result}
    except Exception as exc:
        return {"error": "ASR comparison failed: " + type(exc).__name__}


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
