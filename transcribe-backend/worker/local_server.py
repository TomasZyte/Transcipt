"""
Локальный воркер для разработки БЕЗ облака (экономный вариант на своём GPU).

Имитирует async-интерфейс RunPod Serverless (/run + /status/{id}), поэтому
API-сервер работает с локальным и облачным воркером через ОДИН код диспатчера.
Запуск: uvicorn local_server:app --host 0.0.0.0 --port 8001
"""

import uuid
import threading

from fastapi import FastAPI
from pydantic import BaseModel

from pipeline import process
from handler import _download

app = FastAPI(title="Local GPU worker (RunPod-совместимый)")
_JOBS = {}  # id -> {"status": ..., "output"/"error": ...}


class RunReq(BaseModel):
    input: dict


def _run_job(job_id: str, inp: dict):
    _JOBS[job_id]["status"] = "IN_PROGRESS"
    media_path = None
    try:
        media_path = _download(inp["media_url"])
        out = process(media_path, options=inp.get("options", {}),
                      file_name=inp.get("file_name", "Запись"),
                      file_type=inp.get("file_type", "audio"))
        _JOBS[job_id].update(status="COMPLETED", output=out)
    except Exception as e:
        _JOBS[job_id].update(status="FAILED", error=str(e))
    finally:
        if media_path:
            import os
            try:
                os.remove(media_path)
            except OSError:
                pass


@app.post("/run")
def run(req: RunReq):
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {"status": "IN_QUEUE"}
    threading.Thread(target=_run_job, args=(job_id, req.input), daemon=True).start()
    return {"id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        return {"status": "NOT_FOUND"}
    return {"id": job_id, **job}
