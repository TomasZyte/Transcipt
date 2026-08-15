"""
Точка входа RunPod Serverless.

Три вида задач, различаются по input.action:

  action = "transcribe" (по умолчанию)
      {"media_url": "<presigned URL>"}        — файл из нашего S3
      ИЛИ
      {"source_url": "https://youtu.be/..."}  — ссылка, качаем через yt-dlp
      + {"options": {...}, "file_name": "...", "file_type": "audio|video",
         "max_duration_sec": 7200}            — необязательный потолок длительности
      → pipeline.process() → JSON с segments/speakers/summary

  action = "analyze"
      {"text": "...", "task": "protocol|tasks|numbers|highlights|summary|qa|translate",
       "target_lang": "...", "question": "...", "glossary": "..."}
      → pipeline.analyze() по готовому тексту, без скачивания
      → {"result": "<текст>"}

  action = "probe"
      {"source_url": "..."} → метаданные ссылки без скачивания медиа
      → {"title": ..., "duration": ..., "extractor": ...}

RunPod считает задачу проваленной, если хендлер вернул словарь с ключом "error".
"""

import glob
import json
import os
import subprocess
import tempfile
import traceback
import urllib.request

import runpod
from pipeline import process

try:
    from pipeline import analyze as _analyze
except ImportError:                      # pragma: no cover
    _analyze = None

# Куки для сайтов, которые не отдают контент дата-центрам (в первую очередь YouTube).
# Кладём файл в формате Netscape и указываем путь в переменной окружения.
YTDLP_COOKIES = os.getenv("YTDLP_COOKIES", "")


def _download(url: str) -> str:
    suffix = os.path.splitext(url.split("?")[0])[1] or ".bin"
    path = tempfile.mktemp(suffix=suffix)
    urllib.request.urlretrieve(url, path)
    return path


def _ydl_opts(outtmpl: str) -> dict:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,          # ссылка на видео внутри плейлиста = одно видео
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "socket_timeout": 30,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",   # ffmpeg в образе уже есть — им же режем аудио
        }],
    }
    if YTDLP_COOKIES and os.path.exists(YTDLP_COOKIES):
        opts["cookiefile"] = YTDLP_COOKIES
    return opts


def _probe_link(url: str) -> dict:
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True,
                           **({"cookiefile": YTDLP_COOKIES}
                              if YTDLP_COOKIES and os.path.exists(YTDLP_COOKIES) else {})}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title") or "Запись",
        "duration": int(info.get("duration") or 0),
        "extractor": info.get("extractor_key") or "",
        "uploader": info.get("uploader") or "",
    }


def _download_link(url: str, max_duration_sec: int = 0) -> tuple:
    """Качает аудиодорожку по ссылке. Возвращает (путь_к_файлу, название)."""
    import yt_dlp

    base = tempfile.mktemp()
    with yt_dlp.YoutubeDL(_ydl_opts(base + ".%(ext)s")) as ydl:
        info = ydl.extract_info(url, download=False)

        dur = int(info.get("duration") or 0)
        if max_duration_sec and dur > max_duration_sec:
            raise ValueError(
                f"Запись длиннее допустимого: {dur // 60} мин при лимите "
                f"{max_duration_sec // 60} мин"
            )
        ydl.download([url])

    found = sorted(glob.glob(base + ".*"))
    if not found:
        raise RuntimeError("yt-dlp не создал файл — возможно, площадка не отдала контент")
    return found[0], (info.get("title") or "Запись")


def _media_duration(path: str) -> float:
    """Длительность через ffprobe — мгновенно и до того, как мы потратим GPU.
    Это и есть защита от обхода лимита тарифа файлом-документом: Telegram не
    сообщает боту длительность документа, а ffprobe читает её из контейнера."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", path],
            capture_output=True, check=True, timeout=60,
        ).stdout
        return float(json.loads(out).get("format", {}).get("duration") or 0)
    except Exception as e:
        print(f"[ffprobe] не удалось прочитать длительность: {e}")
        return 0.0


def _handle_analyze(inp: dict) -> dict:
    if _analyze is None:
        return {"error": "analyze не реализован в pipeline.py — обнови образ воркера"}

    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "text отсутствует: нечего анализировать"}

    return {"result": _analyze(
        text,
        task=(inp.get("task") or "summary").strip(),
        target_lang=inp.get("target_lang", "") or "",
        question=inp.get("question", "") or "",
        glossary=inp.get("glossary", "") or "",
    )}


def _handle_transcribe(inp: dict) -> dict:
    media_url = inp.get("media_url")
    source_url = inp.get("source_url")
    if not media_url and not source_url:
        return {"error": "нужен media_url или source_url"}

    file_name = inp.get("file_name", "Запись")
    cleanup = []

    try:
        max_dur = int(inp.get("max_duration_sec") or 0)

        if source_url:
            media_path, title = _download_link(source_url, max_dur)
            file_name = inp.get("file_name") or f"{title[:80]}.m4a"
        else:
            media_path = _download(media_url)
        cleanup.append(media_path)

        if max_dur:
            dur = _media_duration(media_path)
            if dur and dur > max_dur:
                return {"error": f"Запись длиннее допустимого: {int(dur // 60)} мин "
                                 f"при лимите {max_dur // 60} мин"}

        out = process(
            media_path,
            options=inp.get("options", {}) or {},
            file_name=file_name,
            file_type=inp.get("file_type", "audio"),
        )
        if source_url:
            out["sourceUrl"] = source_url
        return out
    finally:
        for p in cleanup:
            try:
                os.remove(p)
            except OSError:
                pass


def handler(job):
    inp = job.get("input", {}) or {}
    action = (inp.get("action") or "transcribe").strip().lower()

    try:
        if action == "analyze":
            return _handle_analyze(inp)
        if action == "probe":
            url = inp.get("source_url")
            if not url:
                return {"error": "source_url отсутствует"}
            return _probe_link(url)
        if action in ("transcribe", ""):
            return _handle_transcribe(inp)
        return {"error": f"неизвестный action: {action}"}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})