"""
Точка входа RunPod Serverless.

RunPod сам выступает очередью и автоскейлером: POST /run кладёт задачу в очередь,
поднимает воркер (scale-to-zero -> холодный старт ~15-30с), гасит его после idle.
Здесь мы только: скачиваем файл по ссылке -> pipeline.process() -> возвращаем JSON.
"""

import os
import tempfile
import urllib.request

import runpod
from pipeline import process


def _download(url: str) -> str:
    suffix = os.path.splitext(url.split("?")[0])[1] or ".bin"
    path = tempfile.mktemp(suffix=suffix)
    urllib.request.urlretrieve(url, path)
    return path


def handler(job):
    """
    Ожидаемый job["input"]:
      { "media_url": "<presigned URL>",
        "options": {...},
        "file_name": "...", "file_type": "audio|video" }
    """
    inp = job.get("input", {})
    media_url = inp.get("media_url")
    if not media_url:
        return {"error": "media_url отсутствует"}

    media_path = _download(media_url)
    try:
        return process(
            media_path,
            options=inp.get("options", {}),
            file_name=inp.get("file_name", "Запись"),
            file_type=inp.get("file_type", "audio"),
        )
    finally:
        try:
            os.remove(media_path)
        except OSError:
            pass


runpod.serverless.start({"handler": handler})
