"""
Диспатчер задач на GPU-воркер. ОДИН код для локального воркера и RunPod Serverless —
отличается только базовый URL и (для RunPod) заголовок авторизации.

  Локально:  DISPATCH_BASE_URL=http://worker:8001
  RunPod:    DISPATCH_BASE_URL=https://api.runpod.ai/v2/<ENDPOINT_ID>
             DISPATCH_API_KEY=<RUNPOD_API_KEY>

Интерфейс намеренно повторяет async-эндпоинты RunPod (/run, /status/{id}),
поэтому переключение "локально <-> облако" — это смена двух переменных окружения.
"""

import os
import httpx

BASE_URL = os.getenv("DISPATCH_BASE_URL", "http://localhost:8001").rstrip("/")
API_KEY = os.getenv("DISPATCH_API_KEY")

_headers = {"Content-Type": "application/json"}
if API_KEY:
    _headers["Authorization"] = f"Bearer {API_KEY}"


def submit(job_input: dict) -> str:
    """Ставит задачу в очередь воркера, возвращает его job id."""
    r = httpx.post(f"{BASE_URL}/run", json={"input": job_input},
                   headers=_headers, timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def status(worker_job_id: str) -> dict:
    """Нормализованный статус: {'state': queued|running|done|failed, 'output'/'error'}."""
    r = httpx.get(f"{BASE_URL}/status/{worker_job_id}", headers=_headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    raw = (data.get("status") or "").upper()
    mapping = {
        "IN_QUEUE": "queued", "IN_PROGRESS": "running",
        "COMPLETED": "done", "FAILED": "failed", "NOT_FOUND": "failed",
    }
    return {
        "state": mapping.get(raw, "running"),
        "output": data.get("output"),
        "error": data.get("error"),
    }
