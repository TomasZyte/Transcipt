"""Restricted, one-message integration entry point. Importing this file makes no API calls."""
import argparse
import asyncio
import hashlib
import logging
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

import httpx

from integration_config import TestSettings, MAX_BYTES, SESSION_SECONDS
from integration_guard import (OneJobRunner, PrivateTestOnly, OutboundTestOnly,
                               SecretFormatter, guarded_payload, permitted_update, process_lock, save_json)


def audited_upload(original, settings, message_id):
    def upload(path, filename=None):
        local = Path(path).resolve(strict=True)
        if not local.is_relative_to(settings.runtime / "tmp") or local.stat().st_size > MAX_BYTES:
            raise RuntimeError("Unexpected test upload path or file size")
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        url = original(path, filename)
        guarded_payload({"media_url": url}, settings)  # Validate destination before GET.
        response = httpx.get(url, timeout=30, follow_redirects=False)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != digest:
            raise RuntimeError("R2 signed download does not match the uploaded file")
        key = unquote(urlsplit(url).path).removeprefix("/" + settings.bucket).lstrip("/")
        save_json(settings.runtime / f"r2-{message_id}.json", {
            "bucket": settings.bucket, "object_key": key, "sha256": digest,
            "bytes": len(response.content), "presigned_get_verified": True})
        return url
    return upload


async def verify_identity(bot, settings):
    identity = await bot.get_me()
    if identity.id != settings.bot_id or (identity.username or "").lower() != settings.username:
        raise RuntimeError("Test Telegram identity mismatch")
    if (await bot.get_webhook_info()).url:
        raise RuntimeError("Test bot has a webhook; refusing to replace it")


async def pending_media(bot, settings):
    await verify_identity(bot, settings)
    # No offset: do not acknowledge or discard pending updates. No file IDs or text in output.
    updates = await bot.get_updates(timeout=0)
    result = []
    for update in updates:
        if not permitted_update(update, settings.user_id) or not update.message:
            continue
        for kind in ("voice", "audio", "video", "video_note", "document"):
            media = getattr(update.message, kind)
            if media:
                result.append({"message_id": update.message.message_id, "type": kind,
                               "bytes": media.file_size, "seconds": getattr(media, "duration", None)})
                break
    return result


async def list_pending(settings):
    from aiogram import Bot
    from aiogram.client.session.aiohttp import AiohttpSession
    import json
    bot = Bot(settings.secrets["BOT_TOKEN"], session=AiohttpSession(timeout=30))
    try:
        print(json.dumps(await pending_media(bot, settings), ensure_ascii=False))
    finally:
        await bot.session.close()


async def poll_once(dp, bot, stop, *, session_seconds=SESSION_SECONDS):
    polling = asyncio.create_task(dp.start_polling(
        bot, handle_as_tasks=False, polling_timeout=5, allowed_updates=["message", "callback_query"],
        handle_signals=False, close_bot_session=False))
    stopped = asyncio.create_task(stop.wait())
    try:
        done, _ = await asyncio.wait({polling, stopped}, timeout=session_seconds,
                                     return_when=asyncio.FIRST_COMPLETED)
        if polling in done:
            await polling
        if not done:
            raise TimeoutError("Test session limit reached")
    finally:
        # Cancellation propagates into the inline handler and its RunPod cleanup.
        polling.cancel()
        stopped.cancel()
        await asyncio.gather(polling, stopped, return_exceptions=True)


async def run(settings, message_id):
    # Only the dedicated CLI imports application modules after config selection.
    import bot as app
    import config
    import db
    import storage
    from aiogram import Bot
    from aiogram.client.session.aiohttp import AiohttpSession
    from aiogram.types import FSInputFile
    from botocore.config import Config

    if config.TEST_SETTINGS != settings:
        raise RuntimeError("Application config was loaded before selecting test mode")
    # bot.py installs its normal formatter on import; restore the stronger test redaction.
    for handler in logging.getLogger().handlers:
        handler.setFormatter(SecretFormatter(settings.secrets.values()))
    runner = OneJobRunner(settings, message_id)
    stop = asyncio.Event()
    # All overrides live in this process; the ordinary bot entry point is unchanged.
    app.TMP_DIR = settings.runtime / "tmp"
    for plan in config.PLANS.values():
        plan["max_file_min"] = 1
    app.rp._run_and_wait = runner
    app.dp.update.outer_middleware(PrivateTestOnly(settings, runner, stop))
    storage._s3 = storage.boto3.client(
        "s3", endpoint_url=config.S3_ENDPOINT, aws_access_key_id=config.S3_ACCESS_KEY,
        aws_secret_access_key=config.S3_SECRET_KEY, region_name="auto",
        config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=20, retries={"max_attempts": 0}))
    storage.upload_file = audited_upload(storage.upload_file, settings, message_id)
    session = AiohttpSession(timeout=60)
    session.middleware(OutboundTestOnly(settings.user_id))
    bot = Bot(config.BOT_TOKEN, session=session)
    try:
        runner.check_available()
        await verify_identity(bot, settings)
        await db.init()
        save_json(settings.runtime / "process.json", {"status": "running", "pid": os.getpid(),
                  "allowed_user_id": settings.user_id, "message_id": message_id})
        logging.info("Test bot ready for one selected message; session limit %s seconds", SESSION_SECONDS)
        await poll_once(app.dp, bot, stop)
        if runner.output is None:
            raise RuntimeError("Selected message did not produce a successful test result")
        transcript_path = settings.runtime / f"transcript-{message_id}.txt"
        transcript_path.write_text(app.exporters.to_txt(runner.output), encoding="utf-8")
        sent = await bot.send_document(settings.user_id, FSInputFile(transcript_path),
                                       caption="Тестовая расшифровка с разметкой спикеров.")
        runner.state.update(transcript_message_id=sent.message_id, telegram_delivery_verified=True)
        save_json(runner.path, runner.state)
    finally:
        try:
            await runner.cancel()
        finally:
            try:
                await bot.session.close()
            finally:
                if db._db:
                    await db._db.close()
                save_json(settings.runtime / "process.json", {"status": "stopped", "pid": os.getpid(),
                          "message_id": message_id, "job_status": runner.state["status"]})
                logging.info("Test bot stopped. Verify cloud jobs and restore test Max workers to 0.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--message-id", type=int)
    mode.add_argument("--check-config", action="store_true", help="Validate locally; no network, DB, polling or GPU")
    mode.add_argument("--list-pending", action="store_true", help="Read pending private media metadata; no acknowledgement or GPU")
    args = parser.parse_args()
    try:
        settings = TestSettings.load(args.env_file)
    except Exception as exc:
        parser.error("Invalid test configuration (" + type(exc).__name__ + "); see integration.env.example")
    if args.check_config:
        print("Test configuration valid; no API requests made")
        return 0
    if not args.list_pending and (not args.message_id or args.message_id <= 0):
        parser.error("--message-id must be a positive ID from the allowed private chat")
    os.environ["POSLOVNICA_TEST_CONFIG"] = str(args.env_file.resolve())
    formatter = SecretFormatter(settings.secrets.values())
    logging.basicConfig(level=logging.INFO)
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        with process_lock(settings.runtime):
            file_handler = logging.FileHandler(settings.runtime / "runner.log", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)
            try:
                asyncio.run(list_pending(settings) if args.list_pending else run(settings, args.message_id))
            finally:
                logging.getLogger().removeHandler(file_handler)
                file_handler.close()
    except KeyboardInterrupt:
        logging.warning("Test run interrupted; inspect remote job status and Max workers")
        return 130
    except Exception as exc:
        # No raw response bodies, credentials or signed URLs in the CLI error path.
        logging.error("Test run failed (%s). Inspect the private journal and test endpoint.", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
