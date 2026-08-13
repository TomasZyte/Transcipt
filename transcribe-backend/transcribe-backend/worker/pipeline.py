"""
Ядро обработки: ffmpeg -> WhisperX (ASR + выравнивание + диаризация) -> LLM-саммари.

Этот модуль НЕ зависит от того, где он запущен (RunPod serverless или локальный
воркер). Он принимает путь к медиафайлу + опции и возвращает результат ровно в той
схеме, которую уже ожидает фронтенд (TranscriptSegment[] / SummaryResult).

Модели грузятся один раз на уровне модуля ("холодный старт") и переиспользуются
между задачами — это и держит горячий GPU быстрым.
"""

import os
import json
import subprocess
import tempfile
from typing import Optional

# --- Конфигурация через окружение -------------------------------------------
DEVICE = os.getenv("DEVICE", "cuda")               # cuda | cpu
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16") # float16 (GPU) | int8 (экономно/CPU)
ASR_MODEL = os.getenv("ASR_MODEL", "large-v3")
HF_TOKEN = os.getenv("HF_TOKEN")                   # токен HuggingFace для pyannote-диаризации
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

# LLM для саммари — любой OpenAI-совместимый endpoint (Ollama, vLLM, TGI, ...)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

# --- Ленивая загрузка тяжёлых моделей ---------------------------------------
_asr_model = None
_align_cache = {}          # language_code -> (model, metadata)
_diarize_pipeline = None


def _load_asr():
    global _asr_model
    if _asr_model is None:
        import whisperx
        _asr_model = whisperx.load_model(ASR_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
    return _asr_model


def _load_align(language_code: str):
    import whisperx
    if language_code not in _align_cache:
        _align_cache[language_code] = whisperx.load_align_model(
            language_code=language_code, device=DEVICE
        )
    return _align_cache[language_code]


def _load_diarizer():
    global _diarize_pipeline
    if _diarize_pipeline is None:
        # Путь импорта менялся между версиями WhisperX — поддерживаем оба.
        try:
            from whisperx.diarize import DiarizationPipeline
        except ImportError:
            from whisperx import DiarizationPipeline
        _diarize_pipeline = DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
    return _diarize_pipeline


# --- Утилиты -----------------------------------------------------------------
def extract_audio(media_path: str) -> str:
    """Любое аудио/видео -> WAV 16кГц mono через ffmpeg. Надёжнее, чем декодер браузера."""
    out_path = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", media_path, "-ar", "16000", "-ac", "1",
         "-c:a", "pcm_s16le", out_path],
        check=True, capture_output=True,
    )
    return out_path


def _fmt_timestamp(total_sec: float) -> str:
    m = int(total_sec // 60)
    s = int(total_sec % 60)
    return f"{m:02d}:{s:02d}"


def _fmt_srt_time(total_sec: float) -> str:
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    ms = int((total_sec - int(total_sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(segments) -> str:
    out = []
    for i, seg in enumerate(segments, 1):
        out.append(f"{i}\n{_fmt_srt_time(seg['startSec'])} --> {_fmt_srt_time(seg['endSec'])}\n"
                   f"[{seg['speaker']}]: {seg['text']}\n")
    return "\n".join(out)


# --- LLM-саммари -------------------------------------------------------------
def summarize(raw_text: str, glossary: str = "") -> dict:
    """Структурированное AI-саммари через локальную LLM (OpenAI-совместимый API)."""
    fallback = {
        "overview": "Расшифровка выполнена успешно.",
        "keyPoints": [], "actionItems": [],
        "sentiment": "Нейтральный / Деловой", "topics": [],
    }
    if not raw_text.strip():
        return fallback

    from openai import OpenAI
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    system = (
        "Ты — аналитик расшифровок встреч. Верни СТРОГО JSON с полями: "
        "overview (строка), keyPoints (массив строк), actionItems (массив строк), "
        "sentiment (строка), topics (массив строк). Отвечай на языке транскрипта."
    )
    user = (
        (f"Глоссарий/термины: {glossary}\n\n" if glossary else "")
        + "Проанализируй расшифрованный текст и составь саммари:\n\n\"\"\"\n"
        + raw_text[:60000] + "\n\"\"\""
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(resp.choices[0].message.content)
        return {**fallback, **data}
    except Exception as e:  # LLM недоступна — не валим всю задачу
        print(f"[summarize] LLM error: {e}")
        return fallback


# --- Главный конвейер --------------------------------------------------------
def process(media_path: str, options: Optional[dict] = None,
            file_name: str = "Запись", file_type: str = "audio") -> dict:
    import whisperx
    options = options or {}
    lang_opt = options.get("language", "auto")
    enable_diar = options.get("enableDiarization", True)
    speaker_count = options.get("speakerCount")  # ТЕПЕРЬ используется (в текущем коде терялось)
    glossary = options.get("customGlossary", "")

    audio_path = extract_audio(media_path)
    audio = whisperx.load_audio(audio_path)

    # 1) Транскрибация
    asr = _load_asr()
    lang = None if lang_opt in (None, "", "auto") else lang_opt
    result = asr.transcribe(audio, batch_size=BATCH_SIZE, language=lang)
    language = result.get("language", "ru")

    # 2) Выравнивание -> точные пословные тайм-коды
    try:
        align_model, metadata = _load_align(language)
        result = whisperx.align(result["segments"], align_model, metadata, audio,
                                DEVICE, return_char_alignments=False)
    except Exception as e:
        print(f"[align] пропущено ({e})")

    # 3) Диаризация ПО ВСЕМУ ФАЙЛУ -> спикеры консистентны от начала до конца
    if enable_diar and HF_TOKEN:
        try:
            diarizer = _load_diarizer()
            diar_kwargs = {}
            if speaker_count:
                diar_kwargs["min_speakers"] = speaker_count
                diar_kwargs["max_speakers"] = speaker_count
            diarize_segments = diarizer(audio, **diar_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            print(f"[diarize] пропущено ({e})")

    # 4) Глобальная перенумерация SPEAKER_00 -> "Спикер 1" (единая по всему файлу)
    speaker_map, segments = {}, []
    for i, seg in enumerate(result.get("segments", []), 1):
        raw_spk = seg.get("speaker", "SPEAKER_00")
        if raw_spk not in speaker_map:
            speaker_map[raw_spk] = f"Спикер {len(speaker_map) + 1}"
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        segments.append({
            "id": f"seg-{i}",
            "startSec": round(start, 2),
            "endSec": round(end, 2),
            "timestamp": _fmt_timestamp(start),
            "speaker": speaker_map[raw_spk],
            "text": (seg.get("text") or "").strip(),
        })

    raw_text = "\n\n".join(f"{s['speaker']}: {s['text']}" for s in segments)
    duration = segments[-1]["endSec"] if segments else 0
    word_count = len(raw_text.split())

    # 5) Саммари
    summary = summarize(raw_text, glossary)

    # чистка временного WAV
    try:
        os.remove(audio_path)
    except OSError:
        pass

    return {
        "languageDetected": language,
        "durationEstimateSec": duration,
        "rawText": raw_text,
        "segments": segments,
        "summary": summary,
        "srtContent": _build_srt(segments),
        "wordCount": word_count,
        "fileName": file_name,
        "fileType": file_type,
    }
