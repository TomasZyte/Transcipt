import asyncio
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration_config import TestSettings
from quality_test import QualityRunner
from test_integration_mode import response


def settings(path):
    return TestSettings(123456789, "fixturebot", 1234, "tt17em69to6yua", "poslovnica-test",
                        "0" * 32 + ".r2.cloudflarestorage.com", Path(path), 2,
                        Decimal("2"), Decimal("0.10"), {"RUNPOD_API_KEY": "offline"})


def payload():
    return {"profile": "baseline", "samples": [{"id": "sample", "sha256": "a" * 64,
            "media_url": "https://" + "0" * 32 + ".r2.cloudflarestorage.com/poslovnica-test/quality/sample.wav?key=offline"}]}


class QualityJournal(unittest.TestCase):
    def test_only_authorized_endpoint_and_bucket(self):
        for changed in ({"endpoint": "l20o867nueti1d"}, {"bucket": "transcribe"}, {"endpoint": "othertest123"}):
            with self.assertRaises(ValueError):
                QualityRunner(replace(settings("unused"), **changed), "baseline")

    def test_budget_counts_previous_telegram_jobs(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = settings(folder)
            runner = QualityRunner(cfg, "baseline")
            runner.path.parent.mkdir()
            for message_id in (10, 11):
                (runner.path.parent / f"message-{message_id}.json").write_text(json.dumps({
                    **runner.state, "message_id": message_id, "status": "COMPLETED"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runner.check_available()

    def test_ambiguous_batch_blocks_other_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = settings(folder)
            first = QualityRunner(cfg, "baseline")
            first.path.parent.mkdir()
            first.path.write_text(json.dumps({**first.state, "status": "reserved"}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                QualityRunner(cfg, "community").check_available()

    def test_batch_policy_and_profile_validation(self):
        runner = QualityRunner(settings("unused"), "baseline")
        body = runner.request_body(payload())
        self.assertEqual(body["policy"]["executionTimeout"], 600000)
        other = payload()
        other["profile"] = "community"
        with self.assertRaises(ValueError):
            runner.request_body(other)


class QualitySubmission(unittest.IsolatedAsyncioTestCase):
    async def test_batch_submission_reuses_durable_journal_and_no_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            client = AsyncMock()
            client.__aenter__.return_value = client
            cfg = settings(folder)
            runner = QualityRunner(cfg, "baseline", poll_s=0, client_factory=lambda **kwargs: client)
            async def post(*args, **kwargs):
                recorded = json.loads(runner.path.read_text())
                self.assertEqual(recorded["status"], "reserved")
                self.assertEqual(recorded["reserved_usd"], "0.50")
                return response({"id": "offline-job"})
            client.post.side_effect = post
            client.get.return_value = response({"status": "COMPLETED", "output": {"samples": []}})
            await runner(payload())
            with self.assertRaises(RuntimeError):
                await runner(payload())
            self.assertEqual(client.post.await_count, 1)


if __name__ == "__main__":
    unittest.main()
