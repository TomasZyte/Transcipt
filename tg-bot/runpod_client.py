"""Клиент RunPod Serverless: транскрибация и текстовые инструменты (analyze)."""
import asyncio
import httpx

import config

_headers = {"Authorization": f"Bearer {config.RUNPOD_API_KEY}",
            "Content-Type": "application/json"}


async def _run_and_wait(payload: dict, timeout_s: int = 1800) -> dict:
    async with httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post(f"{config.RUNPOD_BASE}/run", json={"input": payload}, headers=_headers)
        r.raise_for_status()
        job_id = r.json()["id"]
        waited = 0
        while waited < timeout_s:
            await asyncio.sleep(3)
            waited += 3
            s = await cli.get(f"{config.RUNPOD_BASE}/status/{job_id}", headers=_headers)
            s.raise_for_status()
            data = s.json()
            st = data.get("status")
            if st == "COMPLETED":
                return data.get("output", {})
            if st == "FAILED":
                raise RuntimeError(data.get("error", "RunPod job FAILED"))
        raise TimeoutError("RunPod job timeout")


async def transcribe(media_url: str, file_name: str, file_type: str,
                     options: dict) -> dict:
    return await _run_and_wait({
        "media_url": media_url, "file_name": file_name,
        "file_type": file_type, "options": options,
    })


async def analyze(text: str, task: str, target_lang: str = "",
                  question: str = "", glossary: str = "") -> str:
    out = await _run_and_wait({
        "action": "analyze", "text": text, "task": task,
        "target_lang": target_lang, "question": question, "glossary": glossary,
    })
    return out.get("result", "Пусто.")
