"""Two bounded ASR comparisons on the dedicated test endpoint; default is offline.

No Telegram polling. Original integration configuration/journals stay unchanged.
--submit additionally needs a fresh operator preflight after paid-run approval.
"""
import argparse
import asyncio
import base64
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import logging
from pathlib import Path
import sys

import httpx

from integration_config import TestSettings, RESERVE_USD
from integration_guard import OneJobRunner, SecretFormatter, TERMINAL, process_lock, save_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "transcribe-backend" / "worker"))
from asr_comparison import prepare, digest, read_json, SHA
from quality_batch_handler import validate_request

ENDPOINT = "tt17em69to6yua"
BUCKET = "poslovnica-test"
VARIANTS = ("wx_auto", "fw_v3")
CAMPAIGN = "asr-children-v1"
REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
IMAGE = "registry.runpod.net/tomaszyte-transcipt-codex-asr-build-check-transcribe-backend-worker-dockerfile-asr-whisper:2bd8f040f"
EXECUTION_MS, TTL_MS, CLIENT_SECONDS = 120_000, 240_000, 270


def money(value):
    if not isinstance(value, str):
        raise ValueError("Budget amounts must be decimal strings")
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError("Invalid budget amount")
    return result


def history_files(runtime):
    return {p.name: p for p in (runtime / "jobs").glob("*.json")
            if p.name.startswith(("message-", "retry-", "failed-"))}


class AsrRunner(OneJobRunner):
    def __init__(self, settings, variant, **kwargs):
        if settings.endpoint != ENDPOINT or settings.bucket != BUCKET or variant not in VARIANTS:
            raise ValueError("Only the two Whisper variants and dedicated test resources are allowed")
        self.base_settings = settings
        root = settings.runtime / CAMPAIGN
        if root.resolve() != root or (root / "jobs").resolve() != root / "jobs":
            raise ValueError("ASR runtime must not redirect through links")
        self.campaign_path = root / "campaign.json"
        self.campaign = read_json(self.campaign_path)
        self.campaign_sha = digest(self.campaign_path)
        c = self.campaign
        if (c.get("schema") != 1 or c.get("campaign") != CAMPAIGN or c.get("endpoint") != ENDPOINT
                or c.get("bucket") != BUCKET or c.get("variants") != list(VARIANTS)
                or c.get("image") != IMAGE or c.get("model_revision") != REVISION):
            raise ValueError("Unexpected campaign identity")
        budget, prior = money(c["budget_usd"]), money(c["baseline_spend_usd"])
        if not 0 < budget <= min(settings.budget, Decimal("2")) or prior + 2 * RESERVE_USD > budget:
            raise ValueError("Campaign cannot fund both reserved attempts within the total limit")
        # One fixed namespace: selecting another output folder cannot reset the limit.
        isolated = replace(settings, runtime=root, max_jobs=2, budget=budget, prior_spend=prior)
        super().__init__(isolated, "asr-" + variant,
                         timeout_s=CLIENT_SECONDS, **kwargs)
        self.variant = variant
        if digest(c["manifest"]) != c.get("manifest_sha256"):
            raise ValueError("Campaign manifest changed")
        self.plan = prepare(Path(c["manifest"]), c["clip_id"], variant)
        if self.plan["audio_sha256"] != c["audio_sha256"]:
            raise ValueError("Campaign audio changed")
        self.state.update(campaign_sha256=self.campaign_sha, variant=variant,
                          audio_sha256=self.plan["audio_sha256"], asr_validated=False)
        self.validate_history()

    def validate_history(self):
        if digest(self.campaign_path) != self.campaign_sha:
            raise ValueError("Campaign changed after preparation")
        historical = history_files(self.base_settings.runtime)
        expected = self.campaign.get("history_sha256")
        if not isinstance(expected, dict) or not expected or set(historical) != set(expected):
            raise ValueError("Submission history changed; reconcile billing before continuing")
        for name, path in historical.items():
            if digest(path) != expected[name] or read_json(path).get("status") not in TERMINAL:
                raise ValueError("Historical submission changed or is unresolved")
        if any((self.base_settings.runtime / "jobs").glob("*.tmp")):
            raise ValueError("Historical journal contains an incomplete write")

    def check_available(self):
        self.validate_history()
        super().check_available()
        previous = list(self.path.parent.glob("message-*.json"))
        expected_names = {"message-asr-wx_auto.json"} if self.variant == "fw_v3" else set()
        if {p.name for p in previous} != expected_names:
            raise RuntimeError("Run wx_auto once before fw_v3; no repeats or extra jobs")
        for path in previous:
            old = read_json(path)
            if (old.get("campaign_sha256") != self.campaign_sha or old.get("asr_validated") is not True
                    or old.get("status") != "COMPLETED" or old.get("reserved_usd") != str(RESERVE_USD)
                    or old.get("audio_sha256") != self.plan["audio_sha256"]):
                raise RuntimeError("Previous ASR attempt is not a verified matching success")
        if digest(self.plan["audio_file"]) != self.plan["audio_sha256"]:
            raise ValueError("Audio changed after preparation")

    def preflight(self, now=None):
        """Validate an operator UI observation; this is not an admin API query."""
        p = read_json(self.settings.runtime / "preflight.json")
        stamp = datetime.fromisoformat(p["verified_at_utc"].replace("Z", "+00:00"))
        now = now or datetime.now(timezone.utc)
        if stamp.tzinfo is None or not 0 <= (now - stamp).total_seconds() <= 900:
            raise ValueError("Refresh the operator preflight within 15 minutes of submission")
        expected = {"endpoint": ENDPOINT, "image": IMAGE, "paid_run_authorized": True,
                    "active_workers": 0, "max_workers": 1, "gpu_count": 1,
                    "execution_timeout_seconds": 120, "idle_timeout_seconds": 5,
                    "flashboot": False, "asr_execution_enabled": True}
        if any(type(p.get(k)) is not type(v) or p[k] != v for k, v in expected.items()):
            raise ValueError("Operator preflight does not confirm the approved bounded test setup")
        if p.get("approved_variants") != list(VARIANTS):
            raise ValueError("The two variants require explicit approval")
        rate = money(p["gpu_rate_usd_per_hour"])
        if not 0 < rate <= Decimal("1.00"):
            raise ValueError("Verify the selected GPU price before submitting")
        billed = money(p["test_billed_usd"])
        count = len(list(self.path.parent.glob("message-*.json")))
        # Do not add historical reservations again. Keep new reservations even on failure.
        used = max(self.settings.prior_spend + count * RESERVE_USD, billed)
        if used + RESERVE_USD > self.settings.budget:
            raise RuntimeError("Fresh billing plus the next reservation exceeds the budget")
        self.state["preflight_sha256"] = digest(self.settings.runtime / "preflight.json")
        self.state["planning_total_usd"] = str(used + RESERVE_USD)
        return p

    def payload(self, url):
        p = self.plan
        return {"profile": "asr-comparison", "variant": self.variant, "sample": {
            "id": p["clip_id"], "sha256": p["audio_sha256"], "media_url": url,
            "source_start_seconds": p["source_start_seconds"],
            "source_end_seconds": p["source_end_seconds"], "duration_seconds": p["duration_seconds"]}}

    def request_body(self, payload):
        # Exact equality excludes human references, prompts and arbitrary options.
        if payload != self.payload(payload.get("sample", {}).get("media_url")):
            raise ValueError("Only the prepared clip and fixed variant may be sent")
        validate_request({"profile": "baseline", "samples": [payload["sample"]]}, self.settings.storage_host)
        return {"input": payload, "policy": {"executionTimeout": EXECUTION_MS, "ttl": TTL_MS}}

    async def health(self):
        async with asyncio.timeout(20), self.client_factory(timeout=15) as client:
            response = await client.get(self.url + "/health", headers=self.headers)
            response.raise_for_status()
            data = response.json()
        jobs = data.get("jobs", {})
        if any(type(jobs.get(k)) is not int or jobs[k] != 0 for k in ("inProgress", "inQueue")):
            raise RuntimeError("Test endpoint is busy or health is unknown")
        workers = data.get("workers", {})
        # `ready` overlaps lifecycle counts: RunPod returns idle=1, ready=1
        # for the same scaled-down worker. Bound it separately, not as a second GPU.
        keys = ("idle", "initializing", "ready", "running", "throttled", "unhealthy")
        if (not isinstance(workers, dict) or not workers
                or any(type(v) is not int or v < 0 for v in workers.values())
                or any(k not in keys for k in workers)
                or sum(v for k, v in workers.items() if k != "ready") > 1
                or workers.get("ready", 0) > 1 or workers.get("running", 0) != 0
                or workers.get("unhealthy", 0) != 0 or workers.get("throttled", 0) != 0):
            raise RuntimeError("Unexpected worker health; inspect the test endpoint")

    def validate_output(self, result):
        output = result.get("output", {})
        p = self.plan
        expected = {"status": "completed", "variant_id": self.variant, "clip_id": p["clip_id"],
                    "audio_sha256": p["audio_sha256"], "duration_seconds": p["duration_seconds"],
                    "source_start_seconds": p["source_start_seconds"], "source_end_seconds": p["source_end_seconds"],
                    "config": p["config"]}
        if (result.get("profile") != "asr-comparison" or result.get("sample_id") != p["clip_id"]
                or not isinstance(output, dict) or any(output.get(k) != v for k, v in expected.items())
                or not isinstance(output.get("raw_asr_text"), str)
                or not isinstance(output.get("raw_asr_segments"), list)):
            raise ValueError("ASR result does not match the prepared comparison")
        runtime = output.get("runtime", {})
        lock = runtime.get("lock_sha256", "")
        if (runtime.get("revision") != REVISION or not isinstance(lock, str) or not SHA.fullmatch(lock)
                or not isinstance(runtime.get("packages"), dict) or not runtime["packages"]):
            raise ValueError("Result lacks the expected model and package lock")
        if self.variant == "fw_v3":
            first = read_json(self.path.with_name("message-asr-wx_auto.json"))
            if first.get("model_lock_sha256") != lock:
                raise ValueError("The variants ran with different model/package locks")
        self.state.update(asr_validated=True, model_lock_sha256=lock)
        save_json(self.path, self.state)

    async def execute(self):
        # The shared parent lock also excludes the old Telegram/quality runners.
        with process_lock(self.base_settings.runtime):
            self.check_available()
            self.preflight()
            await self.health()
            url = await asyncio.to_thread(upload_audio, self.settings, self.plan)
            self.check_available()
            self.preflight()  # Reject stale approval/config after any slow upload.
            await self.health()
            result = await super().__call__(self.payload(url))
            self.validate_output(result)
            return result

    async def __call__(self, *args, **kwargs):
        raise RuntimeError("Use execute(): direct submission would bypass the ASR preflight")

    def summary(self):
        return {"status": "prepared_not_run", "variant": self.variant,
                "audio_sha256": self.plan["audio_sha256"], "duration_seconds": self.plan["duration_seconds"],
                "endpoint": ENDPOINT, "baseline_usd": str(self.settings.prior_spend),
                "reserve_per_attempt_usd": str(RESERVE_USD), "max_attempts": 2,
                "planning_total_for_two_usd": str(self.settings.prior_spend + 2 * RESERVE_USD),
                "policy": {"executionTimeout": EXECUTION_MS, "ttl": TTL_MS},
                "client_timeout_seconds": CLIENT_SECONDS, "network_calls": 0,
                "reservation_created": False}


def upload_audio(settings, plan):
    import boto3
    from botocore.config import Config
    data = Path(plan["audio_file"]).read_bytes()
    if len(data) > 2 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != plan["audio_sha256"]:
        raise ValueError("Audio changed before upload")
    s3 = boto3.client("s3", endpoint_url="https://" + settings.storage_host,
        aws_access_key_id=settings.secrets["S3_ACCESS_KEY"],
        aws_secret_access_key=settings.secrets["S3_SECRET_KEY"], region_name="auto",
        config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=20,
                      retries={"total_max_attempts": 1}))
    key = "quality/asr-children-v1/" + plan["audio_sha256"] + "/context.wav"
    s3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType="audio/wav",
                  ContentMD5=base64.b64encode(hashlib.md5(data).digest()).decode("ascii"))
    stream = s3.get_object(Bucket=BUCKET, Key=key)["Body"]
    try:
        received = stream.read(2 * 1024 * 1024 + 1)
    finally:
        stream.close()
    if hashlib.sha256(received).hexdigest() != plan["audio_sha256"]:
        raise ValueError("R2 round-trip checksum mismatch")
    return s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=900)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    try:
        settings = TestSettings.load(args.env_file)
        runner = AsrRunner(settings, args.variant)
        # Offline preparation of the second plan is allowed before the first result.
        if not args.submit:
            print(json.dumps(runner.summary()))
            return 0
        logging.basicConfig(level=logging.INFO)
        for handler in logging.getLogger().handlers:
            handler.setFormatter(SecretFormatter(settings.secrets.values()))
        logging.getLogger("httpx").setLevel(logging.WARNING)
        asyncio.run(runner.execute())
        print("Verified private ASR result saved. Inspect billing and return test Max workers to 0.")
        return 0
    except Exception as exc:
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__, "automatic_retry": False}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
