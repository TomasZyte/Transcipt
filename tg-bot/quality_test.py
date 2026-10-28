"""Explicit, bounded batch benchmark using only the existing test R2/RunPod resources.

No Telegram polling or sends. Reuses the integration submission journal, budget,
timeouts and cancellation. --check is local only; --submit explicitly permits I/O.
"""
import argparse
import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
import sys
import wave

import httpx
from integration_config import TestSettings, EXECUTION_MS, TTL_MS
from integration_guard import OneJobRunner, SecretFormatter, process_lock, save_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transcribe-backend" / "worker"))
from quality_batch_handler import validate_request, MAX_BYTES


def load_samples(directory):
    directory = directory.resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
    samples = manifest.get("samples", [])
    if not 1 <= len(samples) <= 5:
        raise ValueError("Expected 1–5 prepared samples")
    total, ids, checked = 0, set(), []
    for row in samples:
        if not isinstance(row.get("id"), str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", row["id"]):
            raise ValueError("Invalid sample ID")
        path = (directory / row["wav"]).resolve(strict=True)
        if path.parent != directory or path.stat().st_size > MAX_BYTES:
            raise ValueError("Unexpected sample path or size")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["wav_sha256"] or row["id"] in ids:
            raise ValueError("Changed or duplicate sample")
        ids.add(row["id"])
        with wave.open(str(path), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2) or not 0 < duration <= 60:
                raise ValueError("Unexpected audio format or duration")
        total += duration
        checked.append((row, path))
    if total > 180:
        raise ValueError("Sample set exceeds 180 seconds")
    return checked


class QualityRunner(OneJobRunner):
    def __init__(self, settings, profile, **kwargs):
        if settings.endpoint != "tt17em69to6yua" or settings.bucket != "poslovnica-test":
            raise ValueError("Use the authorized quality test endpoint and bucket")
        if profile not in ("baseline", "community"):
            raise ValueError("Unexpected quality profile")
        # Non-numeric IDs cannot collide with Telegram messages. The shared
        # message-*.json journal still counts these against the same two-job cap.
        super().__init__(settings, "quality-" + profile, **kwargs)
        self.profile = profile

    def request_body(self, payload):
        validate_request(payload, self.settings.storage_host)
        if payload["profile"] != self.profile:
            raise ValueError("Wrong profile")
        return {"input": payload, "policy": {"executionTimeout": EXECUTION_MS, "ttl": TTL_MS}}


async def run(settings, profile, directory):
    import boto3
    from botocore.config import Config
    checked = load_samples(directory)
    runner = QualityRunner(settings, profile)
    runner.check_available()
    async with httpx.AsyncClient(timeout=20) as client:
        health = await client.get(runner.url + "/health", headers=runner.headers)
        health.raise_for_status()
        jobs = health.json().get("jobs", {})
        if jobs.get("inProgress") != 0 or jobs.get("inQueue") != 0:
            raise RuntimeError("Test endpoint is busy or health is unknown")
    s3 = boto3.client("s3", endpoint_url="https://" + settings.storage_host,
                      aws_access_key_id=settings.secrets["S3_ACCESS_KEY"],
                      aws_secret_access_key=settings.secrets["S3_SECRET_KEY"], region_name="auto",
                      config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=20,
                                    retries={"max_attempts": 0}))
    payload = {"profile": profile, "samples": []}
    for row, path in checked:
        key = "quality/" + row["wav_sha256"] + "/" + path.name
        await asyncio.to_thread(s3.upload_file, str(path), settings.bucket, key)
        url = s3.generate_presigned_url("get_object", Params={"Bucket": settings.bucket, "Key": key}, ExpiresIn=3600)
        payload["samples"].append({"id": row["id"], "sha256": row["wav_sha256"], "media_url": url})
    result = await runner(payload)
    rows = result.get("samples", [])
    expected = {row["id"]: row["wav_sha256"] for row, _ in checked}
    if (result.get("profile") != profile or len(rows) != len(expected)
            or {row.get("id"): row.get("sha256") for row in rows} != expected):
        raise RuntimeError("Benchmark output does not match the selected samples/profile")
    destination = directory / profile
    destination.mkdir(exist_ok=False)
    for row in rows:
        save_json(destination / (row["id"] + ".json"), row["output"])
    save_json(destination / "batch.json", result)
    print("Batch completed; private results saved. Verify billing and return test Max workers to 0.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--profile", choices=("baseline", "community"), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    try:
        settings = TestSettings.load(args.env_file)
        logging.basicConfig(level=logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger("httpx").setLevel(logging.WARNING)
        checked = load_samples(args.samples)
        runner = QualityRunner(settings, args.profile)
        runner.check_available()
        if args.check:
            print(f"Local checks passed: {len(checked)} files; no API calls or paid jobs.")
            return 0
        with process_lock(settings.runtime):
            asyncio.run(run(settings, args.profile, args.samples))
        return 0
    except Exception as exc:
        print("Quality test stopped (" + type(exc).__name__ + "); inspect configuration/private journal before retrying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
