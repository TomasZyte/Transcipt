"""Сборка файлов экспорта из расшифровки (srt/txt/md)."""


def _fmt_srt_time(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60)
    s = int(sec % 60); ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(t: dict) -> str:
    if t.get("srtContent"):
        return t["srtContent"]
    out = []
    for i, seg in enumerate(t.get("segments", []), 1):
        out.append(f"{i}\n{_fmt_srt_time(seg['startSec'])} --> {_fmt_srt_time(seg['endSec'])}\n"
                   f"[{seg.get('speaker','')}]: {seg.get('text','')}\n")
    return "\n".join(out)


def to_txt(t: dict) -> str:
    return t.get("rawText", "")


def to_md(t: dict, file_name: str = "Запись") -> str:
    s = t.get("summary", {}) or {}
    lines = [f"# Расшифровка: {file_name}", ""]
    if s.get("overview"):
        lines += ["## Краткое содержание", s["overview"], ""]
    if s.get("keyPoints"):
        lines += ["## Ключевые тезисы"] + [f"- {x}" for x in s["keyPoints"]] + [""]
    if s.get("actionItems"):
        lines += ["## Задачи и поручения"] + [f"- {x}" for x in s["actionItems"]] + [""]
    if s.get("topics"):
        lines += ["**Темы:** " + ", ".join(s["topics"]), ""]
    lines += ["## Транскрипт по спикерам", ""]
    for seg in t.get("segments", []):
        lines.append(f"**{seg.get('speaker','')}** ({seg.get('timestamp','')}): {seg.get('text','')}")
    return "\n".join(lines)
