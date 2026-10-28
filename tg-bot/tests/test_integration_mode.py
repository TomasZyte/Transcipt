"""Test-mode isolation and failure paths. All external APIs are fakes."""
import asyncio
from dataclasses import replace
from decimal import Decimal
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aiogram.types import Update
from aiogram.methods import SendMessage, SendInvoice, RefundStarPayment
import httpx
import integration_config as ic
from integration_config import TestSettings
from integration_guard import (OneJobRunner, PrivateTestOnly, OutboundTestOnly, SecretFormatter,
                               guarded_payload, permitted_update, process_lock, save_json)
from integration_test import audited_upload, pending_media, poll_once, verify_identity, run


def settings(runtime):
    return TestSettings(123456789, "fixturebot", 1234, "testendpoint123", "fixture-test",
        "a" * 32 + ".r2.cloudflarestorage.com", Path(runtime), 2, Decimal("2"), Decimal("0.10"),
        {"BOT_TOKEN": "123456789:" + "x" * 35, "RUNPOD_API_KEY": "offline-runpod-secret",
         "S3_ACCESS_KEY": "offline-access", "S3_SECRET_KEY": "offline-storage-secret"})


def env_text(**overrides):
    s = settings("unused")
    values = {**s.secrets, "TEST_BOT_ID": str(s.bot_id), "TEST_BOT_USERNAME": s.username,
        "TEST_ALLOWED_USER_ID": str(s.user_id), "RUNPOD_ENDPOINT_ID": s.endpoint,
        "S3_BUCKET": s.bucket, "S3_ENDPOINT": "https://" + s.storage_host,
        "TEST_MAX_JOBS": "2", "TEST_BUDGET_USD": "2", "TEST_PRIOR_SPEND_USD": "0.10"}
    values.update(overrides)
    return "\n".join(f"{k}={v}" for k,v in values.items())


def update(uid=1234, chat_id=None, chat_type="private", message_id=10, **content):
    return Update.model_validate({"update_id":1, "message": {"message_id":message_id, "date":1,
        "from":{"id":uid,"is_bot":False,"first_name":"Test"},
        "chat":{"id":uid if chat_id is None else chat_id,"type":chat_type}, **content}})


def media(**overrides):
    return {"file_id":"fixture", "file_unique_id":"fixture-unique", "duration":20,
            "file_size":100, **overrides}


def response(data=None, status=200):
    return httpx.Response(status, json=data or {}, request=httpx.Request("GET", "https://example.invalid"))


class ConfigurationCases(unittest.TestCase):
    def test_explicit_file_ignores_inherited_credentials_and_dotenv_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"test.env"
            path.write_text(env_text(), encoding="utf-8")
            with patch.dict(os.environ, {"BOT_TOKEN":"production", "S3_BUCKET":"transcribe",
                    "DB_PATH":"production.db", "PROVIDER_TOKEN":"production", "LOCAL_BOT_API":"http://production"}):
                s = TestSettings.load(path)
            self.assertEqual(s.environment()["BOT_TOKEN"], settings(tmp).secrets["BOT_TOKEN"])
            self.assertEqual(s.environment()["PROVIDER_TOKEN"], "")
            self.assertEqual(s.environment()["LOCAL_BOT_API"], "")
            self.assertEqual(Path(s.environment()["DB_PATH"]), s.runtime/"bot.db")
            self.assertNotIn(s.secrets["BOT_TOKEN"], repr(s))
            path.write_text(env_text(RUNPOD_API_KEY="${RUNPOD_API_KEY}"), encoding="utf-8")
            with self.assertRaises(ValueError):
                TestSettings.load(path)

    def test_config_rejects_production_incomplete_and_invalid_limits(self):
        cases = [
            {"TEST_BOT_ID":str(ic.PRODUCTION_BOT_ID),"BOT_TOKEN":str(ic.PRODUCTION_BOT_ID)+":"+"x"*35},
            {"TEST_BOT_USERNAME":"PoslovnicaBot"}, {"RUNPOD_ENDPOINT_ID":ic.PRODUCTION_ENDPOINT},
            {"S3_BUCKET":"transcribe"}, {"S3_ENDPOINT":"https://example.com"},
            {"TEST_ALLOWED_USER_ID":""}, {"TEST_ALLOWED_USER_ID":"-1"}, {"BOT_TOKEN":"987654321:"+"x"*35},
            {"RUNPOD_API_KEY":""}, {"DB_PATH":"bot.db"}, {"PROVIDER_TOKEN":"paid"},
            {"STARS_ENABLED":"1"}, {"LOCAL_BOT_API":"http://localhost:8081"},
            {"TEST_MAX_JOBS":"3"}, {"TEST_BUDGET_USD":"NaN"}, {"TEST_BUDGET_USD":"Infinity"},
            {"TEST_BUDGET_USD":"3"}, {"TEST_PRIOR_SPEND_USD":"1.75"}, {"TEST_PRIOR_SPEND_USD":"-1"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"test.env"
            for changed in cases:
                with self.subTest(changed=changed):
                    path.write_text(env_text(**changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        TestSettings.load(path)

    def test_config_module_does_not_discover_dotenv_in_test_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"test.env"
            path.write_text(env_text(), encoding="utf-8")
            code = ("from unittest.mock import patch; "
                    "p=patch('dotenv.load_dotenv',side_effect=AssertionError('implicit dotenv'));p.start(); "
                    "import config; assert config.TEST_SETTINGS.user_id==1234; "
                    "assert not config.PAYMENTS_ENABLED and not config.STARS_ENABLED; "
                    "assert config.S3_BUCKET=='fixture-test'")
            env = {**os.environ, "POSLOVNICA_TEST_CONFIG":str(path), "S3_BUCKET":"transcribe"}
            subprocess.run([sys.executable,"-c",code], cwd=Path(__file__).resolve().parents[1],
                           env=env, check=True, capture_output=True, timeout=20)

    def test_check_config_does_not_create_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"test.env"
            path.write_text(env_text(), encoding="utf-8")
            with patch.object(ic,"ROOT",Path(tmp)):
                import integration_test
                with patch.object(sys,"argv",["integration_test.py","--env-file",str(path),"--check-config"]), \
                     patch("integration_test.asyncio.run") as run:
                    self.assertEqual(integration_test.main(),0)
                run.assert_not_called()
                self.assertFalse((Path(tmp)/".test-runtime").exists())

    def test_lock_is_exclusive_and_failed_acquisition_does_not_remove_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with process_lock(root):
                with self.assertRaises(FileExistsError):
                    with process_lock(root):
                        pass
                self.assertTrue((root/"runner.lock").exists())
            self.assertFalse((root/"runner.lock").exists())

    def test_only_private_allowed_updates_and_free_exports(self):
        self.assertTrue(permitted_update(update(text="/start"),1234))
        for event in (update(uid=999,text="/start"),update(chat_id=-100,chat_type="group",text="/start"),
                      Update(update_id=1),Update.model_validate({"update_id":1,"edited_message":update(text="x").message.model_dump()})):
            self.assertFalse(permitted_update(event,1234))

    def test_payload_rejects_other_buckets_urls_actions_and_forces_policy(self):
        s = settings("unused")
        url = "https://"+s.storage_host+"/fixture-test/tg/a/file.ogg?signature=fake"
        value = guarded_payload({"media_url":url,"max_duration_sec":9999,
                                 "options":{"speakerCount":2,"enableDiarization":False}},s)
        self.assertEqual(value["policy"],{"executionTimeout":600000,"ttl":900000})
        self.assertEqual(value["input"]["max_duration_sec"],60)
        self.assertEqual(value["input"]["options"],{"language":"auto","enableDiarization":True})
        for payload in ({"source_url":url},{"action":"analyze"},{"action":"probe"},
            {"media_url":url.replace("fixture-test","transcribe")},
            {"media_url":url.replace("https:","http:")}, {"media_url":url.replace("/tg/a/","/tg/../")},
            {"media_url":url.replace(s.storage_host,s.storage_host+".evil")},
            {"media_url":url.split("?")[0]}):
            with self.assertRaises(RuntimeError):
                guarded_payload(payload,s)

    def test_log_redacts_all_secrets_and_signed_urls_in_tracebacks(self):
        s = settings("unused")
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(SecretFormatter(s.secrets.values()))
        logger = logging.getLogger("integration-redaction")
        logger.addHandler(handler)
        logger.propagate = False
        try:
            try:
                raise ValueError(" ".join(s.secrets.values())+" https://host/file?signature=private")
            except ValueError:
                logger.exception(s.secrets["BOT_TOKEN"].replace(":","~"))
        finally:
            logger.removeHandler(handler)
        self.assertTrue(stream.getvalue())
        for secret in (*s.secrets.values(),"signature=private",s.secrets["BOT_TOKEN"].replace(":","~")):
            self.assertNotIn(secret,stream.getvalue())


class AsyncIntegrationCases(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = settings(self.tmp.name)
        self.client = AsyncMock()
        self.client.__aenter__.return_value = self.client
        self.client.post.return_value = response({"id":"fixture-job"})
        self.client.get.return_value = response({"status":"COMPLETED","output":{"segments":[{"speaker":"Speaker 1"},{"speaker":"Speaker 2"}]}})
        self.factory = Mock(return_value=self.client)
        self.runner = OneJobRunner(self.s,10,poll_s=0,timeout_s=1,client_factory=self.factory)
        self.payload = {"media_url":"https://"+self.s.storage_host+"/fixture-test/tg/a/sample.wav?signature=fake"}

    async def test_success_persists_policy_result_and_prevents_repeat_across_restart(self):
        output = await self.runner(self.payload)
        self.assertEqual(len(output["segments"]),2)
        self.assertEqual(self.client.post.call_args.kwargs["json"]["policy"]["ttl"],900000)
        stored = json.loads(self.runner.path.read_text())
        self.assertEqual(stored["status"],"COMPLETED")
        again = OneJobRunner(self.s,10,client_factory=self.factory)
        with self.assertRaises(RuntimeError):
            await again(self.payload)
        self.assertEqual(self.client.post.await_count,1)

    async def test_ambiguous_submission_is_persisted_never_retried_and_blocks_other_messages(self):
        self.client.post.side_effect = httpx.ReadTimeout("response lost")
        with self.assertRaises(httpx.ReadTimeout):
            await self.runner(self.payload)
        self.assertEqual(json.loads(self.runner.path.read_text())["status"],"reserved")
        with self.assertRaises(RuntimeError):
            await OneJobRunner(self.s,11,client_factory=self.factory)(self.payload)
        self.assertEqual(self.client.post.await_count,1)

    async def test_job_count_and_cumulative_budget_checked_before_post(self):
        self.runner.path.parent.mkdir()
        for number in (1,2):
            save_json(self.runner.path.parent/f"message-{number}.json",{**self.runner.state,"status":"COMPLETED"})
        with self.assertRaises(RuntimeError):
            await self.runner(self.payload)
        self.factory.assert_not_called()
        with tempfile.TemporaryDirectory() as other:
            poor = OneJobRunner(replace(self.s,runtime=Path(other),prior_spend=Decimal("1.75")),10,client_factory=self.factory)
            with self.assertRaises(RuntimeError):
                await poor(self.payload)
        self.factory.assert_not_called()

    async def test_terminal_errors_stop_and_do_not_cancel_completed_remote_jobs(self):
        for status in ("FAILED","CANCELLED","TIMED_OUT"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                self.client.get.return_value = response({"status":status})
                runner = OneJobRunner(replace(self.s,runtime=Path(tmp)),10,poll_s=0,client_factory=self.factory)
                with self.assertRaises(RuntimeError):
                    await runner(self.payload)
                self.assertEqual(runner.state["status"],status)
        self.assertEqual(self.client.post.await_count,3)

    async def test_invalid_completed_output_does_not_become_a_success(self):
        for output in (None,{"error":"failure"},"bad"):
            with tempfile.TemporaryDirectory() as tmp:
                self.client.get.return_value = response({"status":"COMPLETED","output":output})
                runner=OneJobRunner(replace(self.s,runtime=Path(tmp)),10,poll_s=0,client_factory=self.factory)
                with self.assertRaises(RuntimeError):
                    await runner(self.payload)
                self.assertIsNone(runner.output)
        self.assertEqual(self.client.post.await_count,3)

    async def test_poll_timeout_cancels_known_job_and_preserves_original_timeout(self):
        async def slow_get(*args,**kwargs):
            if self.client.post.await_count == 1:
                await asyncio.Event().wait()
            return response({"status":"CANCELLED"})
        self.client.get.side_effect = slow_get
        self.runner.timeout_s=0.03
        with self.assertRaises(TimeoutError):
            await self.runner(self.payload)
        self.assertEqual(self.client.post.call_args.args[0],self.runner.url+"/cancel/fixture-job")
        self.assertEqual(self.runner.state["status"],"CANCELLED")

    async def test_task_cancellation_cancels_remote_job_and_does_not_swallow_cancel(self):
        entered=asyncio.Event()
        async def slow_get(*args,**kwargs):
            if self.client.post.await_count == 1:
                entered.set()
                await asyncio.Event().wait()
            return response({"status":"CANCELLED"})
        self.client.get.side_effect=slow_get
        task=asyncio.create_task(self.runner(self.payload))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.runner.state["status"],"CANCELLED")

    async def test_cancel_failure_records_unresolved_job_without_masking_original_error(self):
        self.client.get.side_effect=httpx.ReadError("status lost")
        self.client.post.side_effect=[response({"id":"fixture-job"}),httpx.ReadError("cancel failed")]
        with self.assertRaises(httpx.ReadError):
            await self.runner(self.payload)
        self.assertEqual(self.runner.state["cancel_error_type"],"ReadError")
        with self.assertRaises(RuntimeError):
            OneJobRunner(self.s,11).check_available()

    async def test_foreign_updates_admin_commands_and_payments_never_reach_handlers(self):
        stop=asyncio.Event()
        guard=PrivateTestOnly(self.s,self.runner,stop)
        handler=AsyncMock()
        for event in (update(uid=55,voice=media()),update(chat_id=-100,chat_type="group",voice=media()),
                      update(text="/broadcast everyone"),update(text="https://example.invalid"),
                      Update(update_id=1),update(text="/refundstars 1 charge")):
            await guard(handler,event,{})
        handler.assert_not_awaited()
        self.factory.assert_not_called()
        self.assertFalse(stop.is_set())

    async def test_only_selected_media_processed_once_and_stops_even_on_upload_error(self):
        stop=asyncio.Event()
        guard=PrivateTestOnly(self.s,self.runner,stop)
        handler=AsyncMock(side_effect=OSError("upload failed"))
        await guard(handler,update(message_id=9,voice=media()),{})
        with self.assertRaises(OSError):
            await guard(handler,update(voice=media()),{})
        await guard(handler,update(voice=media()),{})
        self.assertTrue(stop.is_set())
        handler.assert_awaited_once()

    async def test_oversize_and_overlong_media_stop_without_download_or_gpu(self):
        from aiogram.types import Message
        for values in ({"file_size":ic.MAX_BYTES+1},{"duration":61},{"file_size":None}):
            guard=PrivateTestOnly(self.s,self.runner,asyncio.Event())
            handler=AsyncMock()
            with patch.object(Message,"answer",AsyncMock()):
                await guard(handler,update(voice=media(**values)),{})
            handler.assert_not_awaited()
            self.assertTrue(guard.stop.is_set())

    async def test_outbound_and_payment_guards_block_before_network(self):
        guard=OutboundTestOnly(1234)
        send=AsyncMock()
        await guard(send,None,SendMessage(chat_id=1234,text="ok"))
        for method in (SendMessage(chat_id=5678,text="blocked"),
                RefundStarPayment(user_id=1234,telegram_payment_charge_id="fixture"),
                SendInvoice(chat_id=1234,title="x",description="x",payload="x",currency="XTR",prices=[])):
            with self.assertRaises(RuntimeError):
                await guard(send,None,method)
        self.assertEqual(send.await_count,1)

    async def test_identity_or_webhook_mismatch_refuses_start(self):
        bot=AsyncMock()
        bot.get_me.return_value=NS(id=999,username=self.s.username)
        with self.assertRaises(RuntimeError):
            await verify_identity(bot,self.s)
        bot.get_webhook_info.assert_not_awaited()
        bot.get_me.return_value=NS(id=self.s.bot_id,username=self.s.username)
        bot.get_webhook_info.return_value=NS(url="https://example.invalid/webhook")
        with self.assertRaises(RuntimeError):
            await verify_identity(bot,self.s)

    async def test_pending_list_reads_only_allowed_media_without_acknowledgement(self):
        bot=AsyncMock()
        bot.get_me.return_value=NS(id=self.s.bot_id,username=self.s.username)
        bot.get_webhook_info.return_value=NS(url="")
        bot.get_updates.return_value=[update(voice=media()),update(uid=999,voice=media()),
            update(text="private text"),update(chat_id=-100,chat_type="group",voice=media())]
        result=await pending_media(bot,self.s)
        self.assertEqual(result,[{"message_id":10,"type":"voice","bytes":100,"seconds":20}])
        bot.get_updates.assert_awaited_once_with(timeout=0)
        self.factory.assert_not_called()

    async def test_entry_point_wires_guards_saves_result_and_closes_resources(self):
        app=NS(TMP_DIR=None,rp=NS(_run_and_wait=None),dp=Mock(),
               exporters=NS(to_txt=lambda output: "fixture transcript"))
        config=NS(TEST_SETTINGS=self.s,PLANS={"free":{"max_file_min":30}},
                  S3_ENDPOINT="https://"+self.s.storage_host,S3_ACCESS_KEY="fake",
                  S3_SECRET_KEY="fake",BOT_TOKEN=self.s.secrets["BOT_TOKEN"])
        database=NS(init=AsyncMock(),_db=NS(close=AsyncMock()))
        storage=NS(boto3=Mock(),upload_file=Mock())
        bot=AsyncMock()
        bot.get_me.return_value=NS(id=self.s.bot_id,username=self.s.username)
        bot.get_webhook_info.return_value=NS(url="")
        bot.send_document.return_value=NS(message_id=42)
        session=Mock()
        bot.session=NS(close=AsyncMock())
        async def fake_poll(dp, bot, stop):
            self.assertIsInstance(app.dp.update.outer_middleware.call_args.args[0],PrivateTestOnly)
            self.assertEqual(app.TMP_DIR,self.s.runtime/"tmp")
            self.assertEqual(config.PLANS["free"]["max_file_min"],1)
            await app.rp._run_and_wait(self.payload)
        with patch.dict(sys.modules,{"bot":app,"config":config,"db":database,"storage":storage}), \
                patch("aiogram.Bot",return_value=bot), \
                patch("aiogram.client.session.aiohttp.AiohttpSession",return_value=session), \
                patch("integration_test.OneJobRunner",return_value=self.runner), \
                patch("integration_test.poll_once",side_effect=fake_poll):
            await run(self.s,10)
        self.assertIsInstance(session.middleware.call_args.args[0],OutboundTestOnly)
        self.assertTrue(self.runner.state["telegram_delivery_verified"])
        self.assertEqual(bot.send_document.call_args.args[0],self.s.user_id)
        bot.session.close.assert_awaited_once()
        database._db.close.assert_awaited_once()
        self.assertEqual(json.loads((self.s.runtime/"process.json").read_text())["status"],"stopped")

    async def test_session_deadline_cancels_inline_polling_handler(self):
        finished=asyncio.Event()
        async def polling(*args,**kwargs):
            self.assertFalse(kwargs["handle_as_tasks"])
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()
        with self.assertRaises(TimeoutError):
            await poll_once(NS(start_polling=polling),None,asyncio.Event(),session_seconds=0.02)
        self.assertTrue(finished.is_set())

    async def test_upload_guard_checks_local_size_scope_and_signed_download_hash(self):
        root=self.s.runtime/"tmp"
        root.mkdir()
        local=root/"audio.wav"
        local.write_bytes(b"fixture-audio")
        original=Mock(return_value=self.payload["media_url"])
        upload=audited_upload(original,self.s,10)
        with patch("integration_test.httpx.get",return_value=httpx.Response(200,content=local.read_bytes(),request=httpx.Request("GET","https://example.invalid"))):
            self.assertEqual(upload(local),self.payload["media_url"])
        self.assertTrue(json.loads((self.s.runtime/"r2-10.json").read_text())["presigned_get_verified"])
        with patch("integration_test.httpx.get",return_value=httpx.Response(200,content=b"wrong",request=httpx.Request("GET","https://example.invalid"))):
            with self.assertRaises(RuntimeError):
                upload(local)
        outside=self.s.runtime/"outside.wav"
        outside.write_bytes(b"outside")
        calls=original.call_count
        with self.assertRaises(RuntimeError):
            upload(outside)
        self.assertEqual(original.call_count,calls)


if __name__ == "__main__":
    unittest.main()
