"""
Точка входа RunPod Serverless.

Две задачи в одном эндпоинте, различаются по полю input.action:

  action = "transcribe" (по умолчанию, если поля нет)
      {"media_url": "<presigned URL>", "options": {...},
       "file_name": "...", "file_type": "audio|video"}
      → скачиваем файл → pipeline.process() → JSON с segments/speakers/summary

  action = "analyze"
      {"action": "analyze", "text": "...", "task": "protocol|tasks|numbers|qa|translate",
       "target_lang": "...", "question": "...", "glossary": "..."}
      → pipeline.analyze() по готовому тексту, без скачивания файла
      → {"result": "<текст>"}

RunPod считает задачу проваленной, если хендлер вернул словарь с ключом "error",
поэтому все ошибки отдаём в таком виде — бот покажет их в чате и в логе.
"""

import os
import tempfile
import traceback
import urllib.request

import runpod
from pipeline import process

# analyze может отсутствовать в старой версии pipeline.py — не роняем воркер целиком
try:
    from pipeline import analyze as _analyze
except ImportError:                      # pragma: no cover
    _analyze = None


def _download(url: str) -> str:
    suffix = os.path.splitext(url.split("?")[0])[1] or ".bin"
    path = tempfile.mktemp(suffix=suffix)
    urllib.request.urlretrieve(url, path)
    return path


def _handle_analyze(inp: dict) -> dict:
    if _analyze is None:
        return {"error": "analyze не реализован в pipeline.py — обнови образ воркера"}

    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "text отсутствует: нечего анализировать"}

    task = (inp.get("task") or "summary").strip()
    result = _analyze(
        text,
        task=task,
        target_lang=inp.get("target_lang", "") or "",
        question=inp.get("question", "") or "",
        glossary=inp.get("glossary", "") or "",
    )
    return {"result": result}


def _handle_transcribe(inp: dict) -> dict:
    media_url = inp.get("media_url")
    if not media_url:
        return {"error": "media_url отсутствует"}

    media_path = _download(media_url)
    try:
        return process(
            media_path,
            options=inp.get("options", {}) or {},
            file_name=inp.get("file_name", "Запись"),
            file_type=inp.get("file_type", "audio"),
        )
    finally:
        try:
            os.remove(media_path)
        except OSError:
            pass


def handler(job):
    inp = job.get("input", {}) or {}
    action = (inp.get("action") or "transcribe").strip().lower()

    try:
        if action == "analyze":
            return _handle_analyze(inp)
        if action in ("transcribe", ""):
            return _handle_transcribe(inp)
        return {"error": f"неизвестный action: {action}"}
    except Exception as e:
        traceback.print_exc()          # уйдёт в логи RunPod целиком
        return {"error": f"{type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})