"""
API-сервер (лёгкий, CPU). Front-door сервиса:
  POST /api/transcribe   — принять файл, поставить задачу в очередь, вернуть job_id (мгновенно)
  GET  /api/jobs/{id}    — статус задачи; когда готово — результат в схеме фронтенда
  GET  /api/health

GPU он НЕ трогает — только раздаёт работу воркеру. Держится на дешёвом VPS 24/7,
а дорогой GPU включается воркером только под реальную нагрузку.
"""

import os
import json
import redis
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import storage
import dispatch

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
JOB_TTL = int(os.getenv("JOB_TTL", "86400"))  # храним запись о задаче сутки

app = FastAPI(title="Transcribe API")
app.add_middleware(
    CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"], allow_headers=["*"],
)
_r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), options: str = Form("{}")):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Файл больше {MAX_UPLOAD_MB} МБ")

    try:
        opts = json.loads(options)
    except json.JSONDecodeError:
        opts = {}

    name = file.filename or "Запись"
    file_type = "video" if (file.content_type or "").startswith("video/") \
        or name.lower().endswith((".mp4", ".webm", ".mov", ".avi", ".mkv")) else "audio"

    # 1) файл -> хранилище, 2) presigned-ссылка для воркера, 3) в очередь
    key = storage.upload(data, name)
    worker_id = dispatch.submit({
        "media_url": storage.presigned_url(key),
        "options": opts,
        "file_name": name,
        "file_type": file_type,
    })

    import uuid
    job_id = uuid.uuid4().hex
    _r.setex(_job_key(job_id), JOB_TTL, json.dumps({
        "worker_id": worker_id, "state": "queued",
        "file_name": name, "file_type": file_type, "output": None,
    }))
    # пользователь получает job_id МГНОВЕННО и дальше опрашивает статус
    return {"job_id": job_id, "state": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    raw = _r.get(_job_key(job_id))
    if not raw:
        raise HTTPException(404, "Задача не найдена")
    job = json.loads(raw)

    # финальные состояния кэшируем — воркер больше не дёргаем
    if job["state"] in ("done", "failed"):
        return job

    st = dispatch.status(job["worker_id"])
    job["state"] = st["state"]
    if st["state"] == "done":
        job["output"] = st["output"]
    elif st["state"] == "failed":
        job["error"] = st.get("error", "Ошибка обработки")
    _r.setex(_job_key(job_id), JOB_TTL, json.dumps(job))
    return job
