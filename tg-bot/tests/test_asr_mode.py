"""Offline checks of spending, no-repeat semantics and ASR result provenance."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch
import wave

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import asr_test as a
from integration_config import TestSettings
from integration_guard import process_lock, save_json


def response(data, status=200):
    return httpx.Response(status, json=data, request=httpx.Request("GET", "https://example.invalid"))


def fixture(root):
    settings = TestSettings(123456789, "fixturebot", 1234, a.ENDPOINT, a.BUCKET,
        "0" * 32 + ".r2.cloudflarestorage.com", root, 2, Decimal("2"), Decimal("0.10"),
        {"RUNPOD_API_KEY": "offline-key", "S3_ACCESS_KEY": "offline-id", "S3_SECRET_KEY": "offline-secret"})
    audio = root / "context.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 16000)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"clips": [{"id": "context", "file": audio.name,
        "sha256": a.digest(audio), "source_start_seconds": 8.5, "source_end_seconds": 9.5}]}))
    (root / "jobs").mkdir()
    journal = root / "jobs/message-old.json"
    journal.write_text(json.dumps({"status": "COMPLETED", "reserved_usd": "2.00"}))
    target = root / a.CAMPAIGN
    target.mkdir()
    campaign = {"schema": 1, "campaign": a.CAMPAIGN, "endpoint": a.ENDPOINT, "bucket": a.BUCKET,
        "variants": list(a.VARIANTS), "image": a.IMAGE, "model_revision": a.REVISION,
        "budget_usd": "2", "baseline_spend_usd": "0.116", "manifest": str(manifest),
        "manifest_sha256": a.digest(manifest), "clip_id": "context", "audio_sha256": a.digest(audio),
        "history_sha256": {journal.name: a.digest(journal)}}
    save_json(target / "campaign.json", campaign)
    preflight = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "endpoint": a.ENDPOINT,
        "image": a.IMAGE, "paid_run_authorized": True, "approved_variants": list(a.VARIANTS),
        "active_workers": 0, "max_workers": 1, "gpu_count": 1, "execution_timeout_seconds": 120,
        "idle_timeout_seconds": 5, "flashboot": False, "asr_execution_enabled": True,
        "gpu_rate_usd_per_hour": "0.75", "test_billed_usd": "0.075"}
    save_json(target / "preflight.json", preflight)
    return settings


def output(runner, lock="b" * 64):
    result = deepcopy(runner.plan)
    result.update(status="completed", raw_asr_text="unprompted fixture", raw_asr_segments=[],
        runtime={"revision": a.REVISION, "lock_sha256": lock, "packages": {"faster-whisper": "1.2.1"}})
    result.pop("audio_file")
    return {"profile": "asr-comparison", "sample_id": "context", "output": result}


class AsrOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.settings = fixture(self.root)
        self.runner = a.AsrRunner(self.settings, "wx_auto")

    def test_default_command_is_offline_and_creates_no_reservation(self):
        for variant in a.VARIANTS:
            with patch.object(sys, "argv", ["asr_test.py", "--env-file", "offline.env", "--variant", variant]), \
                 patch.object(a.TestSettings, "load", return_value=self.settings), \
                 patch.object(a, "upload_audio", side_effect=AssertionError("R2 forbidden")), \
                 patch.object(a.asyncio, "run", side_effect=AssertionError("network forbidden")), \
                 patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(a.main(), 0)
                summary = json.loads(stdout.getvalue())
                self.assertEqual(summary["planning_total_for_two_usd"], "1.116")
                self.assertFalse(summary["reservation_created"])
        self.assertFalse(self.runner.path.parent.exists())
        self.assertFalse((self.root / "runner.lock").exists())

    def test_production_other_resources_and_extra_variants_rejected(self):
        for changed in ({"endpoint": "l20o867nueti1d"}, {"endpoint": "differenttest1"}, {"bucket": "transcribe"}):
            with self.assertRaises(ValueError):
                a.AsrRunner(replace(self.settings, **changed), "wx_auto")
        for variant in ("wx_ru", "gigaam_v3", "../bad"):
            with self.assertRaises(ValueError):
                a.AsrRunner(self.settings, variant)

    def test_historical_reserves_not_added_again_and_old_files_untouched(self):
        before = (self.root / "jobs/message-old.json").read_bytes()
        self.runner.check_available()
        self.runner.preflight()
        self.assertEqual(self.runner.state["planning_total_usd"], "0.616")
        self.assertEqual((self.root / "jobs/message-old.json").read_bytes(), before)

    def test_new_or_changed_old_submission_blocks_campaign(self):
        old = self.root / "jobs/message-old.json"
        old.write_text('{"status":"FAILED"}')
        with self.assertRaises(ValueError):
            self.runner.check_available()
        new = self.root / "jobs/retry-extra.json"
        new.write_text('{"status":"COMPLETED"}')
        with self.assertRaises(ValueError):
            a.AsrRunner(self.settings, "wx_auto")

    def test_campaign_and_manifest_cannot_change_silently(self):
        c = a.read_json(self.runner.campaign_path)
        c["baseline_spend_usd"] = "0"
        save_json(self.runner.campaign_path, c)
        with self.assertRaises(ValueError):
            self.runner.check_available()
        (self.root / "manifest.json").write_text('{"clips":[]}')
        with self.assertRaises(ValueError):
            a.AsrRunner(self.settings, "wx_auto")

    def test_missing_stale_future_and_unapproved_preflight_rejected(self):
        path = self.runner.settings.runtime / "preflight.json"
        original = a.read_json(path)
        for changes in ({"paid_run_authorized": False}, {"asr_execution_enabled": False},
                {"max_workers": 2}, {"active_workers": 1}, {"max_workers": True},
                {"image": "wrong"}, {"execution_timeout_seconds": 600}, {"flashboot": True},
                {"gpu_rate_usd_per_hour": "1.01"}, {"gpu_rate_usd_per_hour": "NaN"},
                {"test_billed_usd": "1.51"}, {"approved_variants": ["wx_auto"]},
                {"verified_at_utc": (datetime.now(timezone.utc)-timedelta(minutes=16)).isoformat()},
                {"verified_at_utc": (datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat()}):
            with self.subTest(changes=changes):
                save_json(path, {**original, **changes})
                with self.assertRaises((ValueError, RuntimeError)):
                    self.runner.preflight()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.runner.preflight()

    def test_second_variant_requires_verified_first_result(self):
        second = a.AsrRunner(self.settings, "fw_v3")
        with self.assertRaises(RuntimeError):
            second.check_available()
        self.runner.path.parent.mkdir()
        save_json(self.runner.path, {**self.runner.state, "status": "COMPLETED"})
        with self.assertRaises(RuntimeError):
            second.check_available()

    def test_request_has_only_prepared_audio_and_bounded_policy(self):
        url = "https://" + self.settings.storage_host + "/poslovnica-test/quality/context.wav?fake=1"
        body = self.runner.request_body(self.runner.payload(url))
        self.assertEqual(body["policy"], {"executionTimeout": 120000, "ttl": 240000})
        for payload in ({**body["input"], "prompt": "expected words"},
                self.runner.payload(url.replace("poslovnica-test", "transcribe")),
                {**body["input"], "variant": "fw_v3"}):
            with self.assertRaises(ValueError):
                self.runner.request_body(payload)

    def test_r2_upload_download_checksum_and_no_retry(self):
        data = Path(self.runner.plan["audio_file"]).read_bytes()
        storage = Mock()
        storage.get_object.return_value = {"Body": io.BytesIO(data)}
        storage.generate_presigned_url.return_value = "signed"
        with patch("boto3.client", return_value=storage) as factory:
            self.assertEqual(a.upload_audio(self.settings, self.runner.plan), "signed")
            self.assertEqual(factory.call_args.kwargs["config"].retries, {"total_max_attempts": 1})
            self.assertEqual(storage.put_object.call_args.kwargs["Bucket"], a.BUCKET)
            self.assertEqual(storage.put_object.call_args.kwargs["Body"], data)
            storage.get_object.return_value = {"Body": io.BytesIO(b"corrupt")}
            with self.assertRaises(ValueError):
                a.upload_audio(self.settings, self.runner.plan)
            self.assertEqual(storage.generate_presigned_url.call_count, 1)


class AsrCloudFakes(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.settings = fixture(self.root)
        self.client = AsyncMock()
        self.client.__aenter__.return_value = self.client
        self.factory = Mock(return_value=self.client)
        self.runner = a.AsrRunner(self.settings, "wx_auto", poll_s=0, client_factory=self.factory)
        self.active_runner = self.runner
        self.result_override = None
        self.health_value = {"jobs": {"inProgress": 0, "inQueue": 0}, "workers": {"idle": 0, "running": 0}}
        async def get(url, **kwargs):
            if url.endswith("/health"):
                return response(self.health_value)
            return response({"status": "COMPLETED", "output": self.result_override or output(self.active_runner)})
        async def post(url, **kwargs):
            state = a.read_json(self.active_runner.path)
            self.assertEqual(state["status"], "reserved")
            self.assertTrue((self.root / "runner.lock").exists())
            self.assertEqual(state["reserved_usd"], "0.50")
            return response({"id": "fake-" + self.active_runner.variant})
        self.client.get.side_effect = get
        self.client.post.side_effect = post
        url = "https://" + self.settings.storage_host + "/poslovnica-test/quality/context.wav?secret=signature"
        patcher = patch.object(a, "upload_audio", return_value=url)
        self.upload = patcher.start()
        self.addCleanup(patcher.stop)

    async def test_two_jobs_share_audio_lock_budget_and_no_repeat(self):
        historical = (self.root / "jobs/message-old.json").read_bytes()
        await self.runner.execute()
        second = a.AsrRunner(self.settings, "fw_v3", poll_s=0, client_factory=self.factory)
        self.active_runner = second
        await second.execute()
        self.assertEqual(self.client.post.await_count, 2)
        self.assertEqual(a.read_json(second.path)["planning_total_usd"], "1.116")
        for variant in a.VARIANTS:
            with self.assertRaises(RuntimeError):
                await a.AsrRunner(self.settings, variant, client_factory=self.factory).execute()
        self.assertEqual(self.client.post.await_count, 2)
        self.assertEqual((self.root / "jobs/message-old.json").read_bytes(), historical)
        for path in self.runner.path.parent.glob("*.json"):
            self.assertNotIn("secret=signature", path.read_text())

    async def test_ambiguous_post_never_retried_and_blocks_both_variants(self):
        self.client.post.side_effect = httpx.ReadTimeout("unknown submission outcome")
        with self.assertRaises(httpx.ReadTimeout):
            await self.runner.execute()
        self.assertEqual(a.read_json(self.runner.path)["status"], "reserved")
        for variant in a.VARIANTS:
            with self.assertRaises(RuntimeError):
                await a.AsrRunner(self.settings, variant, client_factory=self.factory).execute()
        self.assertEqual(self.client.post.await_count, 1)

    async def test_timeout_cancels_then_verifies_known_job(self):
        self.runner.timeout_s = .03
        cancelled = False
        async def post(url, **kwargs):
            nonlocal cancelled
            if "/cancel/" in url:
                cancelled = True
                return response({"status": "CANCELLED"})
            return response({"id": "fake-job"})
        async def get(url, **kwargs):
            if url.endswith("/health"):
                return response(self.health_value)
            if cancelled:
                return response({"status": "CANCELLED"})
            await asyncio.Event().wait()
        self.client.post.side_effect = post
        self.client.get.side_effect = get
        with self.assertRaises(TimeoutError):
            await self.runner.execute()
        self.assertEqual(a.read_json(self.runner.path)["status"], "CANCELLED")
        self.assertEqual(self.client.post.await_count, 2)

    async def test_cancellation_ack_without_terminal_status_remains_blocked(self):
        self.runner.timeout_s = .03
        async def get(url, **kwargs):
            if url.endswith("/health"):
                return response(self.health_value)
            if self.client.post.await_count == 1:
                await asyncio.Event().wait()
            return response({"status": "IN_PROGRESS"})
        self.client.get.side_effect = get
        self.client.post.side_effect = [response({"id": "fake-job"}), response({"status": "ack"})]
        with self.assertRaises(TimeoutError):
            await self.runner.execute()
        state = a.read_json(self.runner.path)
        self.assertEqual(state["cancel_observed_status"], "IN_PROGRESS")
        with self.assertRaises(RuntimeError):
            await a.AsrRunner(self.settings, "fw_v3", client_factory=self.factory).execute()
        self.assertEqual(self.client.post.await_count, 2)

    async def test_wrong_result_blocks_second_even_if_provider_completed(self):
        self.result_override = output(self.runner)
        self.result_override["output"]["audio_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            await self.runner.execute()
        state = a.read_json(self.runner.path)
        self.assertEqual(state["status"], "COMPLETED")
        self.assertFalse(state["asr_validated"])
        with self.assertRaises(RuntimeError):
            await a.AsrRunner(self.settings, "fw_v3", client_factory=self.factory).execute()
        self.assertEqual(self.client.post.await_count, 1)

    async def test_different_model_package_lock_rejected_on_second(self):
        await self.runner.execute()
        second = a.AsrRunner(self.settings, "fw_v3", poll_s=0, client_factory=self.factory)
        self.active_runner = second
        self.result_override = output(second, lock="c" * 64)
        with self.assertRaises(ValueError):
            await second.execute()
        self.assertFalse(a.read_json(second.path)["asr_validated"])

    async def test_failed_upload_or_busy_health_never_submits(self):
        self.upload.side_effect = ValueError("round-trip failed")
        with self.assertRaises(ValueError):
            await self.runner.execute()
        self.client.post.assert_not_called()
        self.assertFalse(self.runner.path.exists())
        self.upload.side_effect = None
        for health in ({"jobs": {"inProgress": 1, "inQueue": 0}, "workers": {"running": 1}},
                       {"jobs": {}, "workers": {}},
                       {"jobs": {"inProgress": 0, "inQueue": 0}, "workers": {"initializing": 2}}):
            self.health_value = health
            with self.assertRaises(RuntimeError):
                await self.runner.execute()
        self.client.post.assert_not_called()

    async def test_ready_is_not_counted_as_a_second_idle_worker(self):
        self.health_value = {"jobs": {"inProgress": 0, "inQueue": 0}, "workers": {
            "idle": 1, "initializing": 0, "ready": 1, "running": 0,
            "throttled": 0, "unhealthy": 0}}
        await self.runner.execute()
        self.assertTrue(a.read_json(self.runner.path)["asr_validated"])
        self.assertEqual(self.client.post.await_count, 1)

    async def test_ready_does_not_hide_extra_or_unavailable_workers(self):
        for workers in ({"idle": 2, "ready": 1}, {"idle": 1, "ready": 2},
                {"idle": 1, "ready": 1, "initializing": 1},
                {"idle": 0, "ready": 0, "throttled": 1},
                {"idle": 0, "ready": 1, "unhealthy": 1}):
            with self.subTest(workers=workers):
                self.health_value = {"jobs": {"inProgress": 0, "inQueue": 0}, "workers": workers}
                with self.assertRaises(RuntimeError):
                    await self.runner.execute()
        self.client.post.assert_not_called()
        self.upload.assert_not_called()
        self.assertFalse(self.runner.path.exists())

    async def test_missing_approval_and_existing_parent_lock_stop_before_network(self):
        path = self.runner.settings.runtime / "preflight.json"
        p = a.read_json(path)
        save_json(path, {**p, "paid_run_authorized": False})
        with self.assertRaises(ValueError):
            await self.runner.execute()
        self.factory.assert_not_called()
        self.upload.assert_not_called()
        save_json(path, p)
        with process_lock(self.root):
            with self.assertRaises(FileExistsError):
                await self.runner.execute()
            self.assertTrue((self.root / "runner.lock").exists())
        self.factory.assert_not_called()

    async def test_inherited_direct_call_cannot_bypass_preflight(self):
        with self.assertRaises(RuntimeError):
            await self.runner({"profile": "asr-comparison"})
        self.factory.assert_not_called()
        self.upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
