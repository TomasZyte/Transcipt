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

# LLM для саммари — локальная модель в этом же воркере (self-hosted, без внешних вызовов).
# Грузится на тот же GPU рядом с Whisper. Меняется через переменную окружения LLM_MODEL.
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# --- Ленивая загрузка тяжёлых моделей ---------------------------------------
_asr_model = None
_align_cache = {}          # language_code -> (model, metadata)
_diarize_pipeline = None
_llm = None
_llm_tok = None


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


# --- LLM-саммари (локальная модель в этом же воркере) ------------------------
def _load_llm():
    global _llm, _llm_tok
    if _llm is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL)
        _llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, torch_dtype=torch.float16, device_map=DEVICE,
        )
    return _llm, _llm_tok


def summarize(raw_text: str, glossary: str = "") -> dict:
    """Структурированное AI-саммари через локальную LLM (без внешних вызовов)."""
    import re
    fallback = {
        "overview": "Расшифровка выполнена успешно.",
        "keyPoints": [], "actionItems": [],
        "sentiment": "Нейтральный / Деловой", "topics": [],
    }
    if not raw_text.strip():
        return fallback

    try:
        model, tok = _load_llm()
        system = (
            "Ты — аналитик расшифровок встреч. Верни СТРОГО JSON (без пояснений, без markdown) "
            "с полями: overview (строка), keyPoints (массив строк), actionItems (массив строк), "
            "sentiment (строка), topics (массив строк). Отвечай на языке транскрипта."
        )
        user = (
            (f"Глоссарий/термины: {glossary}\n\n" if glossary else "")
            + "Проанализируй расшифрованный текст и составь саммари:\n\n\"\"\"\n"
            + raw_text[:12000] + "\n\"\"\""
        )
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(**inputs, max_new_tokens=700, do_sample=False)
        text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        return {**fallback, **data}
    except Exception as e:  # не валим всю задачу, если саммари не удалось
        print(f"[summarize] LLM error: {e}")
        return fallback


# --- Текстовые инструменты поверх готовой расшифровки ------------------------
# Длина одного куска текста, который отдаём модели за раз. Qwen2.5-3B держит
# большой контекст, но чем длиннее вход — тем дольше генерация и тем хуже
# модель удерживает детали, поэтому режем и собираем результат обратно.
ANALYZE_CHUNK = int(os.getenv("ANALYZE_CHUNK", "9000"))
ANALYZE_MAX_CHUNKS = int(os.getenv("ANALYZE_MAX_CHUNKS", "12"))

_NO_MD = ("Не используй markdown-разметку: ни звёздочек, ни решёток, ни обратных "
          "кавычек. Обычный текст, списки — с символа «•».")

# task -> (описание системы, инструкция, лимит токенов на ответ)
_TASKS = {
    "protocol": (
        "Ты — секретарь совещаний. Составляешь деловые протоколы: сухо, по существу, "
        "без воды и без выдумывания того, чего в записи не было.",
        "Составь протокол встречи по расшифровке. Структура:\n"
        "ТЕМА — одной строкой\n"
        "УЧАСТНИКИ — перечисли спикеров и, если по репликам понятны роли, укажи их\n"
        "ОБСУЖДЕНИЕ — ключевые вопросы и позиции сторон\n"
        "РЕШЕНИЯ — что именно решили\n"
        "ОТКРЫТЫЕ ВОПРОСЫ — что осталось без ответа",
        900,
    ),
    "tasks": (
        "Ты — ассистент руководителя. Вытаскиваешь из разговора конкретные поручения.",
        "Выпиши все задачи и договорённости из расшифровки. По каждой строкой:\n"
        "• что сделать — кто ответственный — к какому сроку\n"
        "Если ответственный или срок не назван, пиши «не назначен» и «срок не указан». "
        "Не придумывай задачи, которых в тексте нет. Если задач нет — так и напиши.",
        700,
    ),
    "numbers": (
        "Ты — финансовый аналитик. Вычленяешь из разговора все числа и суммы.",
        "Выпиши все упомянутые цифры: суммы, проценты, сроки, объёмы, даты, "
        "количества. По каждой строкой:\n"
        "• значение — к чему относится — кто назвал\n"
        "Сохраняй валюту и единицы измерения как в тексте. Ничего не пересчитывай.",
        700,
    ),
    "highlights": (
        "Ты — аналитик расшифровок. Вычленяешь главные мысли и формулируешь их "
        "коротко, так, чтобы каждый тезис был понятен без чтения всей записи.",
        "Выпиши ключевые тезисы разговора — от 5 до 12 пунктов, каждый строкой "
        "с символа «•», одно-два предложения. Порядок — по важности, а не по "
        "хронологии. Без вступления и без выводов в конце, только тезисы.",
        700,
    ),
    "links": (
        "Ты — помощник, который собирает упомянутые в разговоре ресурсы.",
        "Выпиши всё, на что ссылались участники: ссылки, названия сервисов, "
        "документов, компаний, инструментов, книг. По каждой строкой:\n"
        "• название — в каком контексте упомянуто",
        600,
    ),
    "summary": (
        "Ты — аналитик расшифровок. Пишешь плотные конспекты без воды.",
        "Составь краткое содержание записи: суть в двух-трёх предложениях, затем "
        "3–5 главных пунктов списком с символа «•».",
        700,
    ),
}


def _generate(system: str, user: str, max_new_tokens: int) -> str:
    model, tok = _load_llm()
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": f"{system} {_NO_MD}"},
         {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _split(text: str, size: int) -> list:
    """Режем по границам реплик, чтобы не рвать фразу спикера пополам."""
    if len(text) <= size:
        return [text]
    parts, buf = [], ""
    for block in text.split("\n\n"):
        if len(buf) + len(block) + 2 > size and buf:
            parts.append(buf)
            buf = block
        else:
            buf = f"{buf}\n\n{block}" if buf else block
    if buf:
        parts.append(buf)
    return parts


def analyze(text: str, task: str = "summary", target_lang: str = "",
            question: str = "", glossary: str = "") -> str:
    """Инструменты поверх расшифровки: протокол, задачи, цифры, ссылки, вопрос, перевод.

    Длинный текст обрабатывается кусками, затем результаты сводятся вторым проходом.
    Возвращает готовый текст для отправки пользователю (без markdown)."""
    text = (text or "").strip()
    if not text:
        return "Пустая расшифровка — нечего анализировать."

    task = (task or "summary").strip().lower()
    chunks = _split(text, ANALYZE_CHUNK)
    dropped = 0
    if len(chunks) > ANALYZE_MAX_CHUNKS:
        dropped = len(chunks) - ANALYZE_MAX_CHUNKS
        chunks = chunks[:ANALYZE_MAX_CHUNKS]

    gloss = f"Учитывай термины и имена собственные: {glossary}\n\n" if glossary else ""

    try:
        # --- перевод: каждый кусок отдельно, потом просто склеиваем ---
        if task == "translate":
            lang = target_lang or "русский"
            system = (f"Ты — профессиональный переводчик. Переводишь на {lang}, "
                      "сохраняя разметку реплик «Спикер N:» и смысл без отсебятины.")
            out = [_generate(system, f"{gloss}Переведи на {lang}:\n\n{c}",
                             min(1500, len(c) // 2 + 400)) for c in chunks]
            result = "\n\n".join(out)

        # --- вопрос к записи: ищем ответ в каждом куске, затем сводим ---
        elif task == "qa":
            q = (question or "").strip()
            if not q:
                return "Не понял вопрос — задайте его текстом."
            system = ("Ты отвечаешь на вопросы строго по содержанию расшифровки. "
                      "Если ответа в тексте нет — так и говоришь, не додумывая.")
            if len(chunks) == 1:
                result = _generate(system, f"{gloss}Расшифровка:\n\n{chunks[0]}\n\n"
                                           f"Вопрос: {q}", 700)
            else:
                partial = [_generate(system,
                                     f"Фрагмент расшифровки:\n\n{c}\n\nВопрос: {q}\n"
                                     "Если во фрагменте ответа нет — ответь одним словом: НЕТ.",
                                     500) for c in chunks]
                found = [p for p in partial if p.strip().upper() != "НЕТ" and len(p) > 10]
                if not found:
                    result = "В записи нет ответа на этот вопрос."
                else:
                    result = _generate(
                        system,
                        f"Вопрос: {q}\n\nОтветы, найденные в разных частях записи:\n\n"
                        + "\n\n".join(found) + "\n\nДай один связный ответ без повторов.",
                        700,
                    )

        # --- протокол / задачи / цифры / ссылки / конспект ---
        else:
            system, instruction, limit = _TASKS.get(task, _TASKS["summary"])
            if len(chunks) == 1:
                result = _generate(system, f"{gloss}{instruction}\n\nРасшифровка:\n\n{chunks[0]}",
                                   limit)
            else:
                partial = [
                    _generate(system,
                              f"{gloss}{instruction}\n\nЭто фрагмент {i} из {len(chunks)}. "
                              f"Разбери только его:\n\n{c}", limit)
                    for i, c in enumerate(chunks, 1)
                ]
                result = _generate(
                    system,
                    f"{instruction}\n\nНиже — разборы фрагментов одной записи. "
                    "Сведи их в один документ: убери повторы, сохрани все факты, "
                    "соблюдай структуру.\n\n" + "\n\n---\n\n".join(partial),
                    limit + 300,
                )

        if dropped:
            result += (f"\n\n(Обработано {ANALYZE_MAX_CHUNKS} фрагментов записи, "
                       f"ещё {dropped} пропущено из-за длины.)")
        return result.strip() or "Модель вернула пустой ответ. Попробуйте ещё раз."

    except Exception as e:
        print(f"[analyze] error: {type(e).__name__}: {e}")
        raise


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