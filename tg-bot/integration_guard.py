"""One selected message, a durable submission limit, and bounded RunPod cleanup."""
import asyncio
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import re
import time
from urllib.parse import unquote, urlsplit

import httpx
from aiogram import BaseMiddleware
from aiogram.client.session.middlewares.base import BaseRequestMiddleware

from integration_config import MAX_BYTES, MAX_DURATION, EXECUTION_MS, TTL_MS, RESERVE_USD

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    temp.replace(path)


@contextmanager
def process_lock(runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    path = runtime / "runner.lock"
    # Never remove another process's lock, including after a failed acquisition.
    with path.open("x", encoding="utf-8") as file:
        file.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def permitted_update(update, user_id):
    message = update.message
    if message:
        return bool(message.from_user and message.from_user.id == user_id
                    and message.chat.type == "private" and message.chat.id == user_id)
    callback = update.callback_query
    return bool(callback and callback.from_user.id == user_id and callback.message
                and callback.message.chat.type == "private" and callback.message.chat.id == user_id)


def guarded_payload(payload, settings):
    if payload.get("action", "transcribe") != "transcribe" or payload.get("source_url"):
        raise RuntimeError("Test mode allows only one file transcription through R2")
    url = urlsplit(payload.get("media_url", ""))
    path = unquote(url.path)
    valid = (url.netloc == settings.storage_host and path.startswith("/" + settings.bucket + "/tg/")) or (
        url.netloc == settings.bucket + "." + settings.storage_host and path.startswith("/tg/"))
    if url.scheme != "https" or not valid or not url.query or url.fragment or ".." in path.split("/"):
        raise RuntimeError("Media URL is outside the test bucket")
    # Discard unexpected options; the speaker count is deliberately automatic.
    return {"input": {"media_url": payload["media_url"], "file_name": payload.get("file_name", "audio"),
                      "file_type": payload.get("file_type", "audio"), "max_duration_sec": MAX_DURATION,
                      "options": {"language": "auto", "enableDiarization": True}},
            "policy": {"executionTimeout": EXECUTION_MS, "ttl": TTL_MS}}


class SecretFormatter(logging.Formatter):
    def __init__(self, secrets):
        super().__init__("%(asctime)s %(levelname)s %(message)s")
        self.secrets = tuple(value for value in secrets if value)

    def format(self, record):
        text = super().format(record)
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]").replace(secret.replace(":", "~"), "[REDACTED]")
        return re.sub(r"https://\S+\?\S+", "[SIGNED_URL]", text)


class OneJobRunner:
    def __init__(self, settings, message_id, *, timeout_s=880, poll_s=5, client_factory=httpx.AsyncClient):
        self.settings, self.message_id = settings, message_id
        self.timeout_s, self.poll_s, self.client_factory = timeout_s, poll_s, client_factory
        self.path = settings.runtime / "jobs" / f"message-{message_id}.json"
        self.url = "https://api.runpod.ai/v2/" + settings.endpoint
        self.headers = {"Authorization": "Bearer " + settings.secrets["RUNPOD_API_KEY"]}
        self.state = {"message_id": message_id, "endpoint_id": settings.endpoint,
                      "bot_id": settings.bot_id, "bucket": settings.bucket,
                      "job_id": None, "status": "not_submitted", "reserved_usd": str(RESERVE_USD)}
        self.output = None
        self.finished = False

    def check_available(self):
        prior = list(self.path.parent.glob("message-*.json"))
        if self.path.exists() or len(prior) >= self.settings.max_jobs:
            raise RuntimeError("Test submission limit exhausted or message already submitted")
        if self.settings.prior_spend + (len(prior) + 1) * RESERVE_USD > self.settings.budget:
            raise RuntimeError("Test budget allowance exhausted")
        for path in prior:
            data = json.loads(path.read_text(encoding="utf-8"))
            if any(data.get(k) != self.state[k] for k in ("endpoint_id", "bot_id", "bucket")):
                raise RuntimeError("Test journal belongs to different resources")
            if data.get("status") not in TERMINAL:
                raise RuntimeError("An earlier submission is unresolved; inspect RunPod before continuing")

    async def cancel(self):
        if not self.state["job_id"] or self.state["status"] in TERMINAL:
            return
        try:
            async with asyncio.timeout(20), self.client_factory(timeout=15) as client:
                response = await client.post(self.url + "/cancel/" + self.state["job_id"], headers=self.headers)
                self.state["cancel_http_status"] = response.status_code
                response.raise_for_status()
                # Acknowledgement is not proof of completion: retain unresolved state until verified.
                status = await client.get(self.url + "/status/" + self.state["job_id"], headers=self.headers)
                status.raise_for_status()
                observed = status.json().get("status")
                self.state["cancel_observed_status"] = observed
                if observed in TERMINAL:
                    self.state["status"] = observed
        except Exception as exc:
            self.state["cancel_error_type"] = type(exc).__name__
            logging.error("RunPod cancellation not confirmed; inspect the test endpoint (%s)", type(exc).__name__)
        finally:
            save_json(self.path, self.state)

    def request_body(self, payload):
        return guarded_payload(payload, self.settings)

    async def __call__(self, payload, timeout_s=3600):
        body = self.request_body(payload)
        self.check_available()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state.update(status="reserved", submitted_at=time.time())
        # Persist before POST. An ambiguous submission consumes the allowance; never retry it.
        with self.path.open("x", encoding="utf-8") as file:
            json.dump(self.state, file)
            file.flush()
            os.fsync(file.fileno())
        try:
            async with asyncio.timeout(self.timeout_s), self.client_factory(timeout=30) as client:
                response = await client.post(self.url + "/run", json=body, headers=self.headers)
                response.raise_for_status()
                job_id = response.json().get("id")
                if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
                    raise RuntimeError("RunPod returned an invalid job ID; do not resubmit")
                self.state.update(job_id=job_id, status="IN_QUEUE")
                save_json(self.path, self.state)
                while True:
                    await asyncio.sleep(self.poll_s)
                    response = await client.get(self.url + "/status/" + job_id, headers=self.headers)
                    response.raise_for_status()
                    data = response.json()
                    status = data.get("status")
                    if status != self.state["status"]:
                        logging.info("Test RunPod status: %s", status)
                    self.state.update(status=status, elapsed_sec=round(time.time() - self.state["submitted_at"], 1))
                    for key in ("delayTime", "executionTime", "workerId"):
                        if key in data:
                            self.state[key] = data[key]
                    save_json(self.path, self.state)
                    if status == "COMPLETED":
                        output = data.get("output")
                        if not isinstance(output, dict) or output.get("error"):
                            raise RuntimeError("RunPod returned an invalid or error output")
                        self.output = output
                        save_json(self.path.with_name(f"result-{self.message_id}.json"), output)
                        self.state["speaker_labels"] = sorted({s["speaker"] for s in output.get("segments", []) if s.get("speaker")})
                        save_json(self.path, self.state)
                        return output
                    if status in TERMINAL:
                        raise RuntimeError("Test RunPod job " + status)
        except BaseException:
            await self.cancel()
            raise
        finally:
            self.finished = True


class PrivateTestOnly(BaseMiddleware):
    def __init__(self, settings, runner, stop):
        self.settings, self.runner, self.stop = settings, runner, stop
        self.accepted = False

    async def __call__(self, handler, event, data):
        if not permitted_update(event, self.settings.user_id):
            return None
        message = event.message
        if not message:
            # Only free exports of the user's existing results are allowed while waiting.
            callback = event.callback_query
            if re.fullmatch(r"ex:\d+:(?:menu|txt|srt|md)", callback.data or ""):
                return await handler(event, data)
            return await callback.answer("В этом тесте доступны только файл и экспорт.")
        media = next((getattr(message, key) for key in ("voice", "audio", "video", "video_note", "document")
                      if getattr(message, key)), None)
        if media:
            if message.message_id != self.runner.message_id or self.accepted:
                return None
            self.accepted = True
            try:
                if not media.file_size or media.file_size > MAX_BYTES:
                    return await message.answer("Для теста нужен файл с известным размером до 10 МБ.")
                if getattr(media, "duration", 0) > MAX_DURATION:
                    return await message.answer("Для теста нужна запись до 60 секунд.")
                self.runner.check_available()
                return await handler(event, data)
            finally:
                # Also stop after rejection, upload failure, or a handler that swallowed an exception.
                self.stop.set()
        if message.text in ("/start", "/help", "/balance"):
            return await handler(event, data)
        return None  # No payments, admin commands, links, edits, inline queries or AI tools.


class OutboundTestOnly(BaseRequestMiddleware):
    def __init__(self, user_id):
        self.user_id = user_id

    async def __call__(self, make_request, bot, method):
        if method.__class__.__name__ in ("SendInvoice", "RefundStarPayment", "AnswerPreCheckoutQuery"):
            raise RuntimeError("Payments are disabled in test mode")
        chat_id = getattr(method, "chat_id", None)
        if chat_id is not None and str(chat_id) != str(self.user_id):
            raise RuntimeError("Blocked outbound message outside the allowed private chat")
        return await make_request(bot, method)
