"""Offline regressions: temporary SQLite, fake Telegram, mocked RunPod/R2.

Run from the repository root: python -m unittest discover -s tg-bot/tests -v
"""
import asyncio
import html
import io
import logging
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Never read a developer's .env or use real credentials during import.
with patch("dotenv.load_dotenv"), patch.dict(os.environ, {
    "BOT_TOKEN": "123456789:" + "x" * 35,
    "RUNPOD_API_KEY": "offline", "RUNPOD_ENDPOINT_ID": "offline",
    "S3_ENDPOINT": "http://127.0.0.1:9", "S3_ACCESS_KEY": "offline",
    "S3_SECRET_KEY": "offline", "ADMINS": "", "PROVIDER_TOKEN": "offline:TEST:token",
    "LOCAL_BOT_API": "", "AWS_EC2_METADATA_DISABLED": "true", "STARS_ENABLED": "1",
}):
    import bot
    import config
    import db
    import runpod_client as rp


def message(uid=1, text="https://example.invalid/media"):
    return NS(from_user=NS(id=uid, full_name="Test", username="test"), text=text,
              answer=AsyncMock(), answer_document=AsyncMock())


def callback(data, uid=1):
    return NS(data=data, from_user=NS(id=uid), message=message(uid), answer=AsyncMock())


def payment(charge="charge-1", uid=1, currency="RUB", amount=None):
    m = message(uid)
    m.successful_payment = NS(invoice_payload=f"plan:start:{uid}:1", currency=currency,
        total_amount=amount if amount is not None else config.PRICES["start"],
        telegram_payment_charge_id=charge, provider_payment_charge_id="provider",
        order_info=None)
    return m


@asynccontextmanager
async def quiet_working(*args):
    yield None


class DatabaseCases(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        await db.init(str(Path(self.tmp.name) / "test.db"))
        await db.get_user(1)

    async def asyncTearDown(self):
        await db._db.close()
        self.tmp.cleanup()

    async def test_parallel_usage_is_not_lost(self):
        await asyncio.gather(*(db.add_usage(1, 1) for _ in range(20)))
        self.assertEqual((await db.get_user(1))["minutes_used"], 20)

    async def test_payment_rollback_then_retry(self):
        await db._db.execute("CREATE TRIGGER fail_plan BEFORE UPDATE OF plan ON users "
                             "BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        await db._db.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            await bot.paid(payment(), AsyncMock(), AsyncMock())
        self.assertFalse(await db.payment_exists("charge-1"))
        await db._db.execute("DROP TRIGGER fail_plan")
        await db._db.commit()
        await bot.paid(payment(), AsyncMock(), AsyncMock())
        self.assertEqual((await db.get_user(1))["plan"], "start")

    async def test_payments_extend_once_each_and_preserve_month_usage(self):
        await db.add_usage(1, 25)
        now = int(time.time())
        await asyncio.gather(*(bot.paid(payment(c), AsyncMock(), AsyncMock())
                               for c in ["one", "two", "one"]))
        user = await db.get_user(1)
        self.assertAlmostEqual(user["plan_until"], now + 60 * 86400, delta=5)
        self.assertEqual(user["minutes_used"], 25)
        cur = await db._db.execute("SELECT COUNT(*) FROM payments")
        self.assertEqual((await cur.fetchone())[0], 2)

    async def test_foreign_callbacks_and_qa_do_not_disclose_transcript(self):
        jid = await db.save_job(1, "secret.wav", {"rawText": "private text"})
        with patch.object(rp, "analyze", AsyncMock(return_value="private result")) as analyze, \
                patch.object(bot, "working", quiet_working):
            for action in ["ex:txt", "ex:menu", "tr:en", "tr:menu", "t:summary", "t:qa"]:
                prefix, op = action.split(":")
                c = callback(f"{prefix}:{jid}:{op}", uid=2)
                if prefix == "t":
                    await bot.tool_cb(c, AsyncMock())
                elif prefix == "tr":
                    await bot.translate_cb(c)
                else:
                    await bot.export_cb(c)
                c.message.answer_document.assert_not_awaited()
                c.message.answer.assert_not_awaited()
                self.assertTrue(c.answer.call_args.kwargs.get("show_alert"))
            state = AsyncMock()
            state.get_data.return_value = {"job_id": jid}
            await bot.qa_answer(message(2, "What is inside?"), state)
            analyze.assert_not_awaited()

    async def test_owner_can_export(self):
        jid = await db.save_job(1, "own.wav", {"rawText": "my text"})
        c = callback(f"ex:{jid}:txt")
        await bot.export_cb(c)
        c.message.answer_document.assert_awaited_once()
        self.assertIn(b"my text", c.message.answer_document.call_args.args[0].data)

    async def test_stars_are_not_marked_as_provider_test_payment(self):
        await bot.paid(payment(currency="XTR", amount=config.PRICES_XTR["start"]),
                       AsyncMock(), AsyncMock())
        cur = await db._db.execute("SELECT is_test FROM payments")
        self.assertEqual((await cur.fetchone())[0], 0)

    async def test_receipt_send_failure_remains_pending_until_retry(self):
        with patch.object(config, "IS_TEST_PAYMENTS", False):
            await bot.paid(payment(), AsyncMock(), AsyncMock())
        api = AsyncMock()
        api.send_message.side_effect = OSError("offline")
        with patch.object(config, "ADMINS", {1}):
            await bot.receipt_cmd(message(), NS(args="1 https://example.invalid/receipt"), api)
            self.assertEqual(len(await db.pending_receipts()), 1)
            api.send_message.side_effect = None
            await bot.receipt_cmd(message(), NS(args="1 https://example.invalid/receipt"), api)
            self.assertEqual(len(await db.pending_receipts()), 0)

    async def test_stars_button_visible_without_ruble_provider(self):
        m = message()
        with patch.object(config, "PAYMENTS_ENABLED", False):
            await bot.tariffs(m, AsyncMock())
        markup = m.answer.call_args.kwargs["reply_markup"]
        self.assertIsNotNone(markup)
        self.assertTrue(any(b.callback_data.startswith("stars:")
                            for row in markup.inline_keyboard for b in row))

    async def test_result_and_minutes_roll_back_together(self):
        await db._db.execute("CREATE TRIGGER fail_usage BEFORE UPDATE OF minutes_used ON users "
                             "BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        await db._db.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            await db.save_job(1, "file", {"rawText": "text"}, minutes=2)
        cur = await db._db.execute("SELECT COUNT(*) FROM jobs")
        self.assertEqual((await cur.fetchone())[0], 0)
        self.assertEqual((await db.get_user(1))["minutes_used"], 0)

    async def test_month_rollover_and_invalid_usage(self):
        await db.add_usage(1, 50)
        with patch.object(db, "_month", return_value="2099-01"):
            await db.add_usage(1, 3)
            self.assertEqual((await db.get_user(1))["minutes_used"], 3)
            for value in [-1, float("nan"), float("inf")]:
                with self.assertRaises(ValueError):
                    await db.save_job(1, "file", {}, minutes=value)
        cur = await db._db.execute("SELECT COUNT(*) FROM jobs")
        self.assertEqual((await cur.fetchone())[0], 0)

    async def test_one_use_promo_cannot_be_spent_twice(self):
        await db.get_user(2)
        await db.create_promo("ONE", "start", 1, 1)
        results = await asyncio.gather(db.redeem_promo(1, "ONE"), db.redeem_promo(2, "ONE"))
        self.assertEqual(sum(ok for ok, _ in results), 1)

    async def test_same_lifetime_plan_survives_payment(self):
        await db.create_promo("LIFE", "start", 0, 1)
        await db.redeem_promo(1, "LIFE")
        await bot.paid(payment(), AsyncMock(), AsyncMock())
        self.assertEqual((await db.get_user(1))["plan_until"], 0)

    async def test_mismatched_successful_payment_requests_review_without_grant(self):
        with patch.object(bot, "notify_admins", AsyncMock()) as notify:
            await bot.paid(payment(amount=1), AsyncMock(), AsyncMock())
            notify.assert_awaited_once()
        self.assertFalse(await db.payment_exists("charge-1"))
        self.assertEqual((await db.get_user(1))["plan"], "free")

    async def test_file_and_link_share_user_guard_but_other_users_can_work(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def transcribe(*args, **kwargs):
            started.set()
            await release.wait()
            return {"durationEstimateSec": 60, "rawText": "done"}
        with patch.object(bot, "working", quiet_working), \
                patch.object(rp, "transcribe_url", side_effect=transcribe) as worker, \
                patch.object(bot, "fetch_to_disk", AsyncMock()) as download:
            first = asyncio.create_task(bot.on_link(message(1), AsyncMock(), AsyncMock()))
            other = None
            try:
                await asyncio.wait_for(started.wait(), 1)
                rejected = message(1)
                await bot.on_media(rejected, AsyncMock(), AsyncMock())
                download.assert_not_awaited()
                self.assertIn("обрабатывается", rejected.answer.call_args.args[0])
                started.clear()
                other = asyncio.create_task(bot.on_link(message(2), AsyncMock(), AsyncMock()))
                await asyncio.wait_for(started.wait(), 1)
                self.assertEqual(worker.await_count, 2)
            finally:
                release.set()
                await asyncio.gather(*[t for t in (first, other) if t is not None])
            self.assertEqual((await db.get_user(1))["minutes_used"], 1)
            self.assertEqual((await db.get_user(2))["minutes_used"], 1)
            await bot.on_link(message(1), AsyncMock(), AsyncMock())
            self.assertEqual((await db.get_user(1))["minutes_used"], 2)

    async def test_cancelled_handler_releases_user_guard_without_charge(self):
        started = asyncio.Event()
        async def wait_forever(*args, **kwargs):
            started.set()
            await asyncio.Event().wait()
        with patch.object(bot, "working", quiet_working), \
                patch.object(rp, "transcribe_url", side_effect=wait_forever):
            task = asyncio.create_task(bot.on_link(message(), AsyncMock(), AsyncMock()))
            try:
                await asyncio.wait_for(started.wait(), 1)
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
        self.assertEqual((await db.get_user(1))["minutes_used"], 0)
        with patch.object(bot, "working", quiet_working), \
                patch.object(rp, "transcribe_url", AsyncMock(return_value={"durationEstimateSec": 60})):
            await bot.on_link(message(), AsyncMock(), AsyncMock())
        self.assertEqual((await db.get_user(1))["minutes_used"], 1)

    async def test_media_download_upload_result_and_charge(self):
        m, api = message(), AsyncMock()
        m.voice = NS(file_id="voice", duration=60)
        api.get_file.return_value = NS(file_path="voice/file.ogg")
        async def download(src, dst):
            Path(dst).write_bytes(b"offline media fixture")
        api.download_file.side_effect = download
        def upload(path, name):
            self.assertEqual(Path(path).read_bytes(), b"offline media fixture")
            return "https://example.invalid/object"
        tmp = Path(self.tmp.name) / "downloads"
        with patch.object(bot, "TMP_DIR", tmp), patch.object(bot, "working", quiet_working), \
                patch.object(bot.storage, "upload_file", side_effect=upload), \
                patch.object(rp, "transcribe", AsyncMock(return_value={
                    "durationEstimateSec": 60, "rawText": "transcript"})) as worker:
            await bot.on_media(m, api, AsyncMock())
        self.assertEqual(worker.call_args.kwargs["max_duration_sec"], 1800)
        self.assertEqual((await db.get_user(1))["minutes_used"], 1)
        self.assertEqual(list(tmp.iterdir()), [])
        self.assertEqual((await db.get_job(1, 1))["transcript"]["rawText"], "transcript")
        self.assertEqual(m.answer.call_args_list[0].kwargs["parse_mode"], "HTML")

    async def test_worker_failure_does_not_charge_or_save_job(self):
        with patch.object(bot, "working", quiet_working), \
                patch.object(rp, "transcribe_url", AsyncMock(side_effect=RuntimeError("too long"))), \
                self.assertLogs(level="ERROR"):
            await bot.on_link(message(), AsyncMock(), AsyncMock())
        self.assertEqual((await db.get_user(1))["minutes_used"], 0)
        self.assertIsNone(await db.get_job(1, 1))


class HandlerCases(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_payment_method_rejected_before_checkout(self):
        q = NS(id="q", from_user=NS(id=1), currency="XTR", total_amount=500,
               invoice_payload="plan:start:1:1")
        api = AsyncMock()
        with patch.object(config, "STARS_ENABLED", False):
            await bot.pre_checkout(q, api)
        self.assertFalse(api.answer_pre_checkout_query.call_args.kwargs["ok"])

    async def test_malformed_callbacks_are_rejected(self):
        for data in ["ex:bad:txt", "ex:1:exe", "ex:1:txt:extra"]:
            c = callback(data)
            await bot.export_cb(c)
            c.message.answer_document.assert_not_awaited()
            self.assertTrue(c.answer.call_args.kwargs["show_alert"])

    def test_aiogram_still_injects_dependencies_into_guarded_handlers(self):
        from aiogram.dispatcher.event.handler import HandlerObject
        for fn in (bot.on_media, bot.on_link):
            handler = HandlerObject(callback=fn)
            self.assertEqual(handler.params, {"m", "bot", "state"})
            self.assertTrue(handler.awaitable)

    async def test_checkout_rejects_wrong_amount_currency_and_user(self):
        for currency, amount, payload in [
            ("RUB", 1, "plan:start:1:1"), ("USD", 59000, "plan:start:1:1"),
            ("XTR", 1, "plan:start:1:1"), ("RUB", 59000, "plan:start:2:1"),
            ("RUB", 59000, "garbage")]:
            q = NS(id="q", from_user=NS(id=1), currency=currency,
                   total_amount=amount, invoice_payload=payload)
            api = AsyncMock()
            await bot.pre_checkout(q, api)
            self.assertFalse(api.answer_pre_checkout_query.call_args.kwargs["ok"])

    async def test_checkout_accepts_valid_rub_and_stars(self):
        for currency, amount in [("RUB", 59000), ("XTR", 500)]:
            api = AsyncMock()
            q = NS(id="q", from_user=NS(id=1), currency=currency,
                   total_amount=amount, invoice_payload="plan:start:1:1")
            await bot.pre_checkout(q, api)
            self.assertTrue(api.answer_pre_checkout_query.call_args.kwargs["ok"])

    async def test_partial_download_is_removed_on_error_and_cancellation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            for exc in [OSError("download failed"), asyncio.CancelledError()]:
                api = AsyncMock()
                api.get_file.return_value = NS(file_path="audio/file.ogg")
                async def download(src, dst):
                    Path(dst).write_bytes(b"partial")
                    raise exc
                api.download_file.side_effect = download
                with patch.object(bot, "TMP_DIR", Path(tmp)):
                    with self.assertRaises(type(exc)):
                        await bot.fetch_to_disk(api, "file-id")
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_caption_escapes_dynamic_html_and_keeps_tags_complete(self):
        caption = bot.result_caption({"fileName": "<invoice>&.wav",
            "languageDetected": "<ru>", "summary": {"overview": "<script>&" * 1000,
            "keyPoints": ["<task>"]}})
        self.assertNotIn("<invoice>", caption)
        self.assertNotIn("<script>", caption)
        self.assertIn("&lt;invoice&gt;&amp;.wav", caption)
        self.assertEqual(caption.count("<b>"), caption.count("</b>"))
        self.assertLessEqual(len(html.unescape(caption).encode("utf-16-le")) // 2, 4096)

    async def test_error_log_redacts_token_and_local_path_variant(self):
        stream = io.StringIO()
        # Use the formatter installed by the application, including exception text.
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.getLogger().handlers[0].formatter)
        logger = logging.getLogger("regression-redaction")
        logger.addHandler(handler)
        logger.propagate = False
        try:
            try:
                raise OSError(config.BOT_TOKEN.replace(":", "~"))
            except OSError:
                logger.exception("download %s", config.BOT_TOKEN)
        finally:
            logger.removeHandler(handler)
            logger.propagate = True
        self.assertNotIn(config.BOT_TOKEN, stream.getvalue())
        self.assertNotIn(config.BOT_TOKEN.replace(":", "~"), stream.getvalue())

    async def test_runpod_terminal_states_stop_polling(self):
        for status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
            client = AsyncMock()
            client.post.return_value = NS(raise_for_status=lambda: None, json=lambda: {"id": "job"})
            client.get.return_value = NS(raise_for_status=lambda: None, json=lambda: {"status": status})
            client.__aenter__.return_value = client
            with patch.object(rp.httpx, "AsyncClient", return_value=client), \
                    patch.object(rp.asyncio, "sleep", AsyncMock()):
                with self.assertRaises(RuntimeError):
                    await rp._run_and_wait({}, timeout_s=4)
            self.assertEqual(client.get.await_count, 1)

    async def test_runpod_deadline_includes_slow_submission(self):
        client = AsyncMock()
        client.__aenter__.return_value = client
        async def slow(*args, **kwargs):
            await asyncio.Event().wait()
        client.post.side_effect = slow
        with patch.object(rp.httpx, "AsyncClient", return_value=client):
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(rp._run_and_wait({}, timeout_s=0.03), timeout=1)
            self.assertLess(time.monotonic() - started, 0.5)

    async def test_runpod_completed_output_and_worker_errors(self):
        for output in [{"rawText": "hello"}, {"error": "media too long"}, None]:
            client = AsyncMock()
            client.__aenter__.return_value = client
            client.post.return_value = NS(raise_for_status=lambda: None, json=lambda: {"id": "job"})
            client.get.return_value = NS(raise_for_status=lambda: None,
                json=lambda: {"status": "COMPLETED", "output": output})
            with patch.object(rp.httpx, "AsyncClient", return_value=client), \
                    patch.object(rp.asyncio, "sleep", AsyncMock()):
                if output and "rawText" in output:
                    self.assertEqual(await rp._run_and_wait({}), output)
                else:
                    with self.assertRaises(RuntimeError):
                        await rp._run_and_wait({})


if __name__ == "__main__":
    unittest.main()
