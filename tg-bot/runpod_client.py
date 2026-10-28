"""Клиент RunPod Serverless: транскрибация, ссылки и текстовые инструменты."""
import asyncio
import httpx

import config

_headers = {"Authorization": f"Bearer {config.RUNPOD_API_KEY}",
            "Content-Type": "application/json"}


async def _run_and_wait(payload: dict, timeout_s: int = 3600) -> dict:
    # Includes HTTP request time, not just the sleeps between status polls.
    async with asyncio.timeout(timeout_s), httpx.AsyncClient(timeout=60) as cli:
        r = await cli.post(f"{config.RUNPOD_BASE}/run", json={"input": payload}, headers=_headers)
        r.raise_for_status()
        job_id = r.json()["id"]
        while True:
            await asyncio.sleep(3)
            s = await cli.get(f"{config.RUNPOD_BASE}/status/{job_id}", headers=_headers)
            s.raise_for_status()
            data = s.json()
            st = data.get("status")
            if st == "COMPLETED":
                output = data.get("output")
                if not isinstance(output, dict):
                    raise RuntimeError("RunPod returned invalid output")
                if output.get("error"):
                    raise RuntimeError(str(output["error"]))
                return output
            if st in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                raise RuntimeError(data.get("error") or f"RunPod job {st}")


async def transcribe(media_url: str, file_name: str, file_type: str,
                     options: dict, max_duration_sec: int = 0) -> dict:
    """max_duration_sec — потолок длительности. Воркер проверит его через ffprobe
    до запуска распознавания, поэтому обойти лимит файлом-документом не выйдет."""
    return await _run_and_wait({
        "media_url": media_url, "file_name": file_name,
        "file_type": file_type, "options": options,
        "max_duration_sec": max_duration_sec,
    })


async def transcribe_url(source_url: str, options: dict,
                         max_duration_sec: int = 0) -> dict:
    """Расшифровка по ссылке: воркер сам качает через yt-dlp."""
    return await _run_and_wait({
        "action": "transcribe",
        "source_url": source_url,
        "options": options,
        "file_type": "video",
        "max_duration_sec": max_duration_sec,
    })


async def probe(source_url: str) -> dict:
    """Название и длительность ссылки без скачивания медиа."""
    return await _run_and_wait({"action": "probe", "source_url": source_url})


async def analyze(text: str, task: str, target_lang: str = "",
                  question: str = "", glossary: str = "") -> str:
    out = await _run_and_wait({
        "action": "analyze", "text": text, "task": task,
        "target_lang": target_lang, "question": question, "glossary": glossary,
    })
    return out.get("result", "Пусто.")
