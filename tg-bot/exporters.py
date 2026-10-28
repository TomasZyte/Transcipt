"""Сборка файлов экспорта из расшифровки (srt/txt/md)."""
import time

# Подряд идущие реплики одного спикера склеиваются в абзац: Whisper режет речь
# по предложениям, и без склейки файл превращается в столбик из «Спикер 1:».
# Новый абзац начинается, если сменился спикер, если пауза больше GAP секунд
# или если абзац разросся больше MAX_CHARS.
GAP = 20.0
MAX_CHARS = 1200


def _fmt_srt_time(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60)
    s = int(sec % 60); ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_time(sec: float) -> str:
    """MM:SS для коротких записей, H:MM:SS — для длинных."""
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _blocks(segments: list) -> list:
    """[{start, speaker, text}] — склеенные реплики."""
    out = []
    for seg in segments:
        speaker = seg.get("speaker", "")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("startSec", 0.0))
        end = float(seg.get("endSec", start))

        last = out[-1] if out else None
        same = (last and last["speaker"] == speaker
                and start - last["end"] <= GAP
                and len(last["text"]) + len(text) <= MAX_CHARS)
        if same:
            last["text"] = f"{last['text']} {text}"
            last["end"] = end
        else:
            out.append({"start": start, "end": end, "speaker": speaker, "text": text})
    return out


def _header_facts(t: dict) -> list:
    facts = []
    dur = t.get("durationEstimateSec")
    if dur:
        facts.append(f"Длительность: {_fmt_time(dur)}")
    speakers = {s.get("speaker") for s in t.get("segments", []) if s.get("speaker")}
    if speakers:
        facts.append(f"Спикеров: {len(speakers)}")
    if t.get("languageDetected"):
        facts.append(f"Язык: {t['languageDetected']}")
    if t.get("wordCount"):
        facts.append(f"Слов: {t['wordCount']}")
    return facts


def to_srt(t: dict) -> str:
    if t.get("srtContent"):
        return t["srtContent"]
    out = []
    for i, seg in enumerate(t.get("segments", []), 1):
        out.append(f"{i}\n{_fmt_srt_time(seg['startSec'])} --> {_fmt_srt_time(seg['endSec'])}\n"
                   f"[{seg.get('speaker','')}]: {seg.get('text','')}\n")
    return "\n".join(out)


def to_txt(t: dict, file_name: str = "") -> str:
    name = file_name or t.get("fileName") or "Запись"
    lines = [
        f"Расшифровка: {name}",
        "Выгружено: " + time.strftime("%d.%m.%Y %H:%M"),
    ]
    facts = _header_facts(t)
    if facts:
        lines.append(" · ".join(facts))

    s = t.get("summary", {}) or {}
    if s.get("overview"):
        lines += ["", "КРАТКО", s["overview"]]
    if s.get("keyPoints"):
        lines += ["", "ТЕЗИСЫ"] + [f"• {x}" for x in s["keyPoints"]]

    lines += ["", "─" * 60, ""]

    blocks = _blocks(t.get("segments", []))
    if not blocks:
        lines.append(t.get("rawText", ""))
    else:
        for b in blocks:
            lines.append(f"[{_fmt_time(b['start'])}] {b['speaker']}:")
            lines.append(b["text"])
            lines.append("")

    lines += ["", "Расшифровано в @PoslovnicaBot"]
    return "\n".join(lines)


def to_md(t: dict, file_name: str = "Запись") -> str:
    name = file_name or t.get("fileName") or "Запись"
    s = t.get("summary", {}) or {}

    lines = [f"# Расшифровка: {name}", ""]
    facts = _header_facts(t)
    if facts:
        lines += [" · ".join(facts), ""]
    lines += [f"*Выгружено {time.strftime('%d.%m.%Y %H:%M')}*", ""]

    if s.get("overview"):
        lines += ["## Краткое содержание", "", s["overview"], ""]
    if s.get("keyPoints"):
        lines += ["## Ключевые тезисы", ""] + [f"- {x}" for x in s["keyPoints"]] + [""]
    if s.get("actionItems"):
        lines += ["## Задачи и поручения", ""] + [f"- {x}" for x in s["actionItems"]] + [""]
    if s.get("topics"):
        lines += ["**Темы:** " + ", ".join(s["topics"]), ""]

    lines += ["## Транскрипт", ""]
    blocks = _blocks(t.get("segments", []))
    if not blocks:
        lines.append(t.get("rawText", ""))
    else:
        for b in blocks:
            lines.append(f"**[{_fmt_time(b['start'])}] {b['speaker']}**")
            lines.append("")
            lines.append(b["text"])
            lines.append("")

    lines += ["---", "", "*Расшифровано в [@PoslovnicaBot](https://t.me/PoslovnicaBot)*"]
    return "\n".join(lines)