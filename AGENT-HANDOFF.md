# Poslovnica (@PoslovnicaBot): руководство для ИИ-агента

Составлено 27.10.2026, обновлено 28.10.2026 после локальной проверки Draft PR #1. Проверенные изменения см. в разделах 0.1–0.2, `tg-bot-status.md` и `tg-bot/REVIEW-2026-10-28.md`. Источники: документы проекта «Transcribe Service», GitHub `TomasZyte/Transcipt` и предоставленная владельцем копия VPS. Исторические сведения ниже относятся к августу–октябрю 2026; актуальное состояние рабочих сервисов отдельно не проверялось.

> **Прочитай до начала работы.** Ниже — пять расхождений между документацией, git и продакшеном. Если их не учесть, можно затереть рабочий код.

---

## 0. Пять критичных фактов

1. **Код бота в GitHub main устарел.** В `main` в папке `tg-bot/` лежит версия от 13.08 (коммит `22e6bfa`, 330 строк `bot.py`, 4 таблицы БД). Боевая версия работает на VPS в `C:\Projects\Transcipt-main\tg-bot` и умеет гораздо больше: оплату Stars, таблицы `payments`/`settings`/`events`, свой Bot API, ссылки через yt-dlp, `/evaltask`, `/refundstars`, `/receipt`, `FileGoneError`, `_safe_err()`, `backup.ps1`. В опубликованной GitHub main ничего из этого нет; копия VPS опубликована в отдельной ветке.
   → **Production и ветка PR — разные версии.** Импорт `4190641` опубликован в `codex/sync-vps-2026-10-27`, открыт [Draft PR #1](https://github.com/TomasZyte/Transcipt/pull/1). Поверх импорта подготовлены исправления (§0.2); их нельзя считать уже установленными на VPS. Не деплой `tg-bot` из git без отдельного решения владельца и проверки текущего состояния VPS.
2. **Код воркера в GitHub актуален.** RunPod собирает образ из ветки `main`, папка `transcribe-backend/worker`. Любой push в `main` вызывает пересборку продакшен-эндпоинта.
3. **Модели НЕ вшиты в образ**, хотя `project-summary.md` и `tg-bot-status.md` утверждают обратное. Коммит `e2ed2df` (15.08) откатил вшивание: сборка не укладывалась в лимит RunPod в 30 минут. Сейчас при холодном старте качаются Whisper large-v3, модель выравнивания, pyannote и Qwen2.5-3B, всего около 10 ГБ. В Dockerfile есть комментарий с планом: если возвращать вшивание, то только Whisper и выравнивание (~4,5 ГБ), без LLM.
4. **Часть поздних изменений подтверждена копией с VPS.** Потоковая загрузка, клиентский таймаут 3600 с, `/stats`, экспорт, события и social proof уже есть в коде. Остальные статусы «Этапа 0» и текущие настройки сервисов требуют проверки.
5. **Для боевой работы нужен доступ к VPS (Windows), RunPod и R2.** Для локальной синхронизации достаточно GitHub и предоставленной владельцем копии файлов. Доступ к работающим службам, секретам и БД в рамках этой синхронизации не использовался. Команды владельцу давать одной строкой PowerShell.

---

## 0.1. Проверенная локальная синхронизация — 27.10.2026

- Рабочий репозиторий: `C:\Users\Projects\Transcipt-git`; локальная `main` обновлена до GitHub `bd5f963`.
- Ветка импорта: `codex/sync-vps-2026-10-27`.
- Бот получен из предоставленной владельцем папки `C:\Users\Projects\Transcipt-main_from VPS\tg-bot`: семь Python-модулей, вспомогательные текстовые файлы, `backup.ps1`, четыре баннера.
- Исходная папка не изменялась. Секреты, БД, логи, бэкапы и временные файлы не импортировались.
- Воркер взят из GitHub main без изменения его кода и промптов.
- Старые локальные правки `handler.py` и `pipeline.py` сохранены в Git stash и `C:\Users\Projects\Transcribe\sync-backups\2026-10-27-before-sync`. Их ранняя реализация `analyze` не наложена поверх более поздней версии GitHub.
- Сверены SHA-256 всех 15 импортированных файлов, проверен синтаксис 13 Python-файлов без запуска приложения.
- Импорт опубликован коммитом `4190641` в Draft PR #1. Деплой и миграции рабочей БД не выполнялись. Текущее состояние служб и настроек RunPod/R2 не проверено.
- Подробности и подтверждённые выполненные задачи: `tg-bot-status.md` в корне рабочего репозитория. Остальные упомянутые ниже документы `claude/*.md` пока не предоставлены.

---

## 0.2. Локальная проверка и исправления — 28.10.2026

- В той же ветке исправлены проверка владельца задания, атомарность платежа и выдачи тарифа, конкурентный учёт минут, обработка чеков, доступность Stars, очистка частичных загрузок, HTML результата и маскирование токена в логах.
- `save_job(..., minutes=...)` сохраняет результат и расход одной транзакцией. `apply_payment(...)` заменяет раздельные `add_payment`/`extend_plan`: повтор платежа безопасен, расход календарного месяца не сбрасывается.
- Общая SQLite connection защищена от перемешивания операций разных задач. Одновременно допускается одна транскрибация на пользователя (файл или ссылка); это защита одного polling-процесса.
- RunPod: общий таймаут 3600 секунд с учётом HTTP, обработка `FAILED`, `CANCELLED`, `TIMED_OUT` и ошибок в output. Отмены задания в облаке при локальном таймауте и восстановления после рестарта пока нет.
- 28 локальных тестов прошли на Python 3.12.14; `pip check`, синтаксис 14 Python-файлов и `git diff --check` прошли. Внешние API подменены; production не запускался.
- README и `.env.example` обновлены. Добавлен `TG_TIMEOUT` из окружения (по умолчанию 1800). Схема БД, зависимости и код воркера относительно импорта не менялись.
- Полный отчёт, оставшиеся ограничения и план выпуска/отката: [tg-bot/REVIEW-2026-10-28.md](tg-bot/REVIEW-2026-10-28.md). PR остаётся Draft; слияние и деплой не выполнялись.

---

## 1. Что это за сервис

Telegram-бот для расшифровки встреч. Пользователь присылает аудио, видео, голосовое, кружок, документ или ссылку (VK Видео, RuTube, Дзен, прямую). В ответ получает:
- расшифровку с разметкой спикеров и тайм-кодами;
- конспект;
- инструменты по кнопкам: краткое содержание, тезисы, протокол, задачи, цифры, вопрос к AI, перевод на 7 языков;
- экспорт в `.srt`, `.txt` или `.md`.

- Бот: **@PoslovnicaBot**, имя «Poslovnica — расшифровка встреч», id `8831823389`.
- Ранний бренд веб-версии — Tranzip AI.
- Владелец: самозанятый (НПД), работает один. Реквизиты — в `claude/tg-bot-status.md`.
- Метрики на 16.09: около 9 пользователей, 2 платящих. Первая оплата — 15.08.2026, 590 ₽.

### Схема работы

```
Пользователь (Telegram)
   │
   ▼
Свой Telegram Bot API-сервер  ── VPS, служба TelegramBotAPI, :8081, --local, файлы до 2 ГБ
   │  getFile → путь на диске C:\telegram-bot-api\data\...
   ▼
Бот (aiogram 3, Python 3.12)   ── VPS, служба PoslovnicaBot, SQLite bot.db
   │  1) файл → Cloudflare R2 (S3) → presigned URL (TTL 3600)
   │  2) POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run, затем опрос /status каждые 3 с
   ▼
RunPod Serverless, эндпоинт «Transcipt»  ── GPU 24 ГБ, только Ampere (3090/A5000)
   handler.py → pipeline.process():
   ffmpeg → WhisperX large-v3 → alignment → pyannote (диаризация по всему файлу)
   → «Спикер N» → summarize() на локальной Qwen2.5-3B (transformers)
   ▼
JSON {segments, rawText, srtContent, summary, ...} → бот сохраняет в jobs → ответ в чат
```

---

## 2. Где что хранится

### 2.1. Карта мест

| Что | Где | Комментарий |
|---|---|---|
| Репозиторий | GitHub `TomasZyte/Transcipt`, ветка `main` | Публичный. Воркер актуален, бот — нет (см. §0) |
| Локальная копия у владельца | `C:\Users\Projects\Transcipt-git` | Рабочий Git-репозиторий; инструкция также в `C:\Users\Projects\Transcribe` |
| Продакшен-бот | VPS (Windows), `C:\Projects\Transcipt-main\tg-bot` | Источник истины для бота |
| `.env` бота | `C:\Projects\Transcipt-main\tg-bot\.env` | Не в git. Файл скрытый: смотреть через `Get-ChildItem -Force` |
| БД | `...\tg-bot\bot.db` (SQLite) | Не в git |
| Логи бота | `...\tg-bot\logs\bot.log` | |
| Бэкапы БД | `...\tg-bot\backups\`, скрипт `...\tg-bot\backup.ps1` | Задача планировщика `PoslovnicaBotBackup`, каждый день в 4:00, хранятся 14 дней |
| Bot API-сервер | `C:\telegram-bot-api\bin\telegram-bot-api.exe` | Служба `TelegramBotAPI` |
| Данные и логи Bot API | `C:\telegram-bot-api\data\`, `...\data\service.log` | Временные файлы пользователей; бот удаляет их после чтения |
| Файлы для воркера | Cloudflare R2, бакет `transcribe` (по умолчанию), ключи `tg/<uuid>/<filename>` | Автоочистка бакета не зафиксирована — стоит проверить lifecycle |
| GPU-воркер | RunPod → Serverless → эндпоинт «Transcipt» | Собирается из GitHub |
| Секреты воркера | Переменные окружения эндпоинта в консоли RunPod | `HF_TOKEN` и другие, см. §4.2 |
| Документы проекта | Claude Project «Transcribe Service», папка `claude/*.md` | См. §9 |
| Стратегия (полный отчёт) | https://claude.ai/artifact/NuH3HhEiXhF19voEC5Twb3 | |
| Кейс для инвесторов | https://claude.ai/artifact/3ocUn4s3t2J6KX32aVyU2S | |
| Тексты оферты и политики | `oferta-ready.md`, `privacy-ready.md` | Лежат у владельца, в репо и в проекте их нет; не опубликованы |

### 2.2. Дерево репозитория (состояние на коммит `bd5f963`)

```
Transcipt/
├── tg-bot/                      ← Telegram-бот (локально импортирована копия VPS; опубликованный main ещё старый)
│   ├── bot.py                   хендлеры aiogram, main(), polling
│   ├── config.py                env, PLANS, PRICES, LANGS, ADMINS
│   ├── db.py                    SQLite: схема и все запросы
│   ├── keyboards.py             reply/inline-клавиатуры (main_menu, result_kb, buy_kb, export_kb, translate_kb)
│   ├── runpod_client.py         /run + опрос /status; transcribe(), analyze()
│   ├── storage.py               boto3 → R2, upload_bytes() → presigned URL
│   ├── exporters.py             to_srt / to_txt / to_md
│   ├── requirements.txt         aiogram 3.13.1, aiosqlite, httpx, boto3, python-dotenv
│   ├── .env.example
│   └── README.md
│
├── transcribe-backend/
│   ├── worker/                  ← ПРОДАКШЕН-ВОРКЕР RunPod (build context)
│   │   ├── Dockerfile           pytorch 2.4.1-cuda12.1-cudnn9 + ffmpeg + cuDNN 8 в /opt/cudnn8
│   │   ├── requirements.txt     whisperx 3.3.1, transformers 4.46.3, runpod 1.7.7, yt-dlp, bitsandbytes...
│   │   ├── handler.py           точка входа RunPod; action = transcribe | analyze | probe
│   │   ├── pipeline.py          806 строк: ASR, выравнивание, диаризация, LLM, промпты, merge
│   │   └── local_server.py      FastAPI-эмуляция /run и /status для локального запуска без RunPod
│   ├── api/                     ← НЕ ИСПОЛЬЗУЕТСЯ в продакшене (FastAPI + Redis для веб-версии)
│   ├── docker-compose.dev.yml   ← локальный стенд (Redis, MinIO, Ollama, воркер) — legacy
│   ├── .env.example             ← legacy: LLM_BASE_URL/Ollama воркер больше не читает
│   ├── README.md
│   └── RUNPOD_DEPLOY.md
│
├── server.ts, src/, index.html, vite.config.ts, package.json
│                                ← ВЕБ-ВЕРСИЯ на Gemini (Google AI Studio, React + Express) — legacy
├── .env.example                 GEMINI_API_KEY, APP_URL (для веб-версии)
└── .gitignore                   .env, *.db, logs/, backups/ — не коммитить
```

**В продакшене живут только `tg-bot/` (с VPS) и `transcribe-backend/worker/`.** Веб-версия на Gemini, `api/` и `docker-compose` — заготовки под будущий веб-кабинет (по роадмапу — Q2 2027). Без задачи на веб их не трогай.

---

## 3. Компоненты подробно

### 3.1. GPU-воркер (`transcribe-backend/worker/`)

**`handler.py`** — точка входа, `runpod.serverless.start({"handler": handler})`. Если вернуть словарь с ключом `"error"`, RunPod засчитает задачу как FAILED, а бот получит `RuntimeError`.

| `input.action` | Вход | Выход |
|---|---|---|
| `transcribe` (по умолчанию) | `media_url` (presigned R2) **или** `source_url` (ссылка для yt-dlp); `options`, `file_name`, `file_type`, `max_duration_sec` | `languageDetected, durationEstimateSec, rawText, segments[], summary{}, srtContent, wordCount, fileName, fileType[, sourceUrl]` |
| `analyze` | `text`, `task` (`protocol, tasks, numbers, highlights, summary, qa, translate`), `target_lang`, `question`, `glossary` | `{"result": "<текст>"}` |
| `probe` | `source_url` | `{title, duration, extractor, uploader}` без скачивания |

`options` для `transcribe`: `language` (`auto` по умолчанию), `enableDiarization` (`true`), `speakerCount`, `customGlossary`.

Элемент `segments[]`: `{id, startSec, endSec, timestamp, speaker: "Спикер N", text}`.

Защита лимитов: если передан `max_duration_sec`, после скачивания `ffprobe` проверяет длительность. Если запись длиннее, воркер возвращает ошибку до запуска Whisper. Для ссылок длительность сравнивается ещё до скачивания.

**`pipeline.py`** — что где менять:

| Строки (примерно) | Что там | Когда менять |
|---|---|---|
| 20–31 | env: `DEVICE, COMPUTE_TYPE, ASR_MODEL, HF_TOKEN, BATCH_SIZE, LLM_MODEL, LLM_4BIT` | Значения задаются в env RunPod, а не в коде |
| 41–70 | `_load_asr`, `_load_align`, `_load_diarizer` — ленивая загрузка моделей | Замена модели ASR (например, A/B с GigaAM) |
| 71 | `extract_audio` (ffmpeg → wav) | |
| 105 | `_load_llm` — transformers, fp16 или 4-bit (nf4) | Смена LLM |
| 130–225 | `_SUM_SYSTEM`, `summarize()` — конспект при транскрибации (по фрагментам, затем сводка) | Качество конспекта |
| 226–227 | `ANALYZE_CHUNK=9000`, `ANALYZE_MAX_CHUNKS=12` | Размер фрагментов для analyze |
| 229–263 | `_NO_MD`, `_RULES`, `_MERGE` (`union` / `compress` / `structure`) | Общие правила и сборка фрагментов |
| 264–379 | **`_TASKS`** — промпты всех инструментов: `system`, `instruction`, `limit`, `merge` | **Главное место правки промптов** |
| 380–432 | `_dedupe_lines`, `_drop_truncated_tail`, `_generate` | Защита от повторов и обрывов |
| 449 | `_qa_chunks` — «Спросить AI» | |
| 504–643 | Протокол: `_SECTIONS`, `_parse_sections`, `_merge_protocol` (сборка по разделам) | Формат протокола |
| 644 | `analyze()` — диспетчер инструментов | Новый инструмент: добавить ключ в `_TASKS` + кнопку в боте |
| 728 | `process()` — главный конвейер транскрибации | |

**`Dockerfile`** — порядок слоёв важен: apt → cuDNN 8 → requirements → `COPY *.py`. Новый `.py` файл нужно явно добавить в строку `COPY`.

### 3.2. Бот (`tg-bot/`)

Ниже описана версия по документам с уточнениями из копии VPS от 27.10.2026. Ветка PR содержит импорт и исправления от 28.10.2026; опубликованная GitHub main пока содержит старого бота.

- **Стек:** aiogram 3, Python 3.12, пакеты установлены глобально, long-polling.
- **`bot.py`**:
  - сессия `AiohttpSession(api=TelegramAPIServer.from_base(LOCAL_BOT_API, is_local=True))`;
  - `fetch_to_disk()` — получает файл на диск в `tg-bot/tmp/`; для локального Bot API переносит его через `os.replace` / `shutil.move`. После потоковой загрузки в R2 временный файл удаляется; если исходного файла нет, бросает `FileGoneError`;
  - `on_media` — приём медиа;
  - `on_link` — ссылки; зарегистрирован **после** state-хендлеров и **перед** fallback, порядок регистрации важен;
  - `tool_cb` (`t:<job>:<task>`), `translate_cb` (`tr:`), `export_cb` (`ex:`);
  - оплата: `buy` → `send_invoice`, `pre_checkout` (сверяет payload и user_id, ответить нужно за 10 с), `paid`;
  - `_safe_err()` маскирует токен в логах, `events` и админ-уведомлениях; глобальный `@dp.errors()`.
- **Команды пользователя:** `start, help, balance, plans, promo, settings, support`.
- **Команды админа** (id из `ADMINS`):
  - `/newpromo КОД plan months uses` (`months=0` — бессрочно, `uses=-1` — без ограничений);
  - `/refundstars user_id charge_id`;
  - `/receipt <id> <ссылка>`;
  - `/evaltask <режим> <N>` — прогон режима по N записям из `jobs` только через LLM, без повторной транскрибации; присылает `.md`-отчёт;
  - `/stats` — уже реализован; также есть `/export`, `/maintenance`, `/broadcast`.
- **`config.py`**:
  - `PLANS` — `free` 90 мин/мес, файл ≤ 30 мин; `start` 1200 / ≤ 120; `pro` 3600 / ≤ 240;
  - `PRICES` **в копейках** (59000 / 129000);
  - `PRICES_XTR` в звёздах (500 / 1100);
  - `LANGS` — 7 языков перевода;
  - `ADMINS`, `LOCAL_BOT_API`, `STARS_ENABLED`, `SOCIAL_PROOF_MIN`.
- **`db.py`, таблицы:**
  - `users` — `plan`, `plan_until` (0 = бессрочно), `minutes_used`, `period` `YYYY-MM`, `lang`, `glossary`;
  - `jobs` — `transcript` хранится как JSON;
  - `promo`, `redemptions`;
  - `payments` — `charge_id` UNIQUE;
  - `settings`, `events`.
  - Ключевые функции: `effective_plan`, `remaining_minutes`, `save_job(..., minutes=...)` (результат и расход атомарно), `add_usage` (календарный месяц UTC), `apply_payment` (платёж и продление атомарно, бессрочность сохраняется при оплате того же тарифа), `stats()` (фильтрует суммы по валюте).
- **`storage.py`** — `upload_file(path)` уже реализован: multipart по 8 МБ, вызов из бота через `asyncio.to_thread`. `upload_bytes()` сохранён для небольших данных.
- **`runpod_client.py`** — `_run_and_wait` (таймаут 3600 с, опрос каждые 3 с), `transcribe()`, `transcribe_url()`, `probe()`, `analyze()`.
- **`keyboards.py`** — `buy_kb()` пора переделать под переключатель «месяц / год».

### 3.3. Свой Telegram Bot API-сервер

- Собран из `tdlib/telegram-bot-api` под Windows. Зачем: облачный API отдаёт файлы только до 20 МБ, свой — до 2 ГБ.
- На облачном API выполнен `logOut`, поэтому токен работает **только** со своим сервером.
- Откат: `http://localhost:8081/bot<TOKEN>/logOut`, затем убрать `LOCAL_BOT_API` из `.env` и выполнить `Restart-Service PoslovnicaBot`.

---

## 4. Конфигурация и секреты

### 4.1. `.env` бота (`tg-bot\.env`, не в git)

```
BOT_TOKEN            токен @BotFather
RUNPOD_ENDPOINT_ID   id эндпоинта «Transcipt»
RUNPOD_API_KEY
S3_ENDPOINT          https://<account>.r2.cloudflarestorage.com
S3_BUCKET            transcribe
S3_ACCESS_KEY / S3_SECRET_KEY / S3_REGION=auto / PRESIGN_TTL=3600
DB_PATH              bot.db (относительный путь, считается от рабочей папки)
ADMINS               telegram id через запятую
PROVIDER_TOKEN       ЮKassa (магазин 1435788)
LOCAL_BOT_API        http://localhost:8081
STARS_ENABLED
SOCIAL_PROOF_MIN     500
YTDLP_COOKIES        (если используется в боте; в воркере — точно)
```

Подводные камни:
- **инлайн-комментарии ломают python-dotenv** — в боевом `.env` их быть не должно;
- `AppDirectory` службы обязан быть папкой `tg-bot`, иначе не найдутся `.env` и `bot.db`.

### 4.2. Env эндпоинта RunPod

```
HF_TOKEN=<токен HF>           нужен pyannote; без него диаризация молча пропускается
DEVICE=cuda
COMPUTE_TYPE=float16
ASR_MODEL=large-v3
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
LLM_4BIT=0                    1 + Qwen2.5-7B-Instruct → 4-bit через bitsandbytes
BATCH_SIZE=16
ANALYZE_CHUNK=9000, ANALYZE_MAX_CHUNKS=12
YTDLP_COOKIES=<путь к cookies.txt в формате Netscape>   для YouTube; файл должен оказаться в контейнере
```

### 4.3. Настройки эндпоинта RunPod

- Queue based, сборка из GitHub: ветка `main`, Dockerfile `transcribe-backend/worker/Dockerfile`, context `transcribe-backend/worker`.
- GPU: **только RTX 3090 и A5000**. L4 периодически снова появляется в списке включённых — проверять и снимать.
- Max workers 1, Active 0, FlashBoot включён, Idle timeout 60–120 с.
- Execution timeout: по старым документам 1800 с; текущее значение в RunPod не проверено. Для 4-часовых записей требуется проверить и при необходимости поднять до ≥ 3600. В коде `runpod_client` таймаут уже 3600 с.
- Container disk: в документах встречается и 20, и 40 ГБ. Раз модели качаются при старте, нужно **не меньше 40 ГБ**.

---

## 5. Где что изменять — рецепты

| Задача | Файлы и действия | Деплой |
|---|---|---|
| Цена тарифа | `config.py` → `PRICES` (копейки) и `PRICES_XTR` (звёзды); текст в `keyboards.buy_kb()`; тексты «Тарифы» и `/plans` в `bot.py`; Description в BotFather | Перезапуск бота |
| Лимиты тарифа | `config.py` → `PLANS[plan].minutes / max_file_min`. Бот передаёт воркеру `max_duration_sec = min(лимит файла, остаток минут)` | Перезапуск бота |
| Новый тариф (например, годовой) | Сейчас `apply_payment` выдаёт один срок `SUB_DAYS`. Потребуется согласовать срок, добавить его в расчёт, обновить `PLANS`/`PRICES`/`PRICES_XTR`, `invoice_plan`, `buy_kb`, текст счёта и оферту; ключ тарифа без двоеточий | Перезапуск бота после проверки |
| Промокод | Команда `/newpromo` в боте, код не трогать | — |
| Промпт инструмента (протокол, задачи и т. д.) | `pipeline.py` → `_TASKS[...]`. **Сначала `/evaltask` по текущей версии, потом правка, потом `/evaltask` ещё раз** | push в `main` → пересборка RunPod |
| Промпт конспекта при транскрибации | `pipeline.py` → `_SUM_SYSTEM`, `summarize()` | push |
| Новый инструмент | `_TASKS` + ветка в `analyze()` → кнопка в `keyboards.result_kb()` → обработка в `tool_cb` | push + перезапуск бота |
| Язык перевода | `config.LANGS` (бот); промпт `translate` в `_TASKS` | Перезапуск бота |
| Модель ASR или LLM | env RunPod (`ASR_MODEL`, `LLM_MODEL`, `LLM_4BIT`) или `_load_*` в `pipeline.py` | Env — новый релиз эндпоинта; код — push |
| Зависимости воркера | `worker/requirements.txt`. **Не поднимать `transformers` выше 4.46.3 и torch выше 2.4.1 без полной проверки** (§7) | push |
| Новые GPU (4090, 5090, L4) | Требуется torch ≥ 2.5 + совместимые whisperx/transformers/cuDNN → сначала тест на отдельном эндпоинте | Отдельный эндпоинт |
| Формат экспорта | `tg-bot/exporters.py` + `keyboards.export_kb()` + `export_cb` | Перезапуск бота |
| Тексты, аватар, команды бота | @BotFather (шаблоны в `claude/tg-bot-branding.md`). **Убрать фразу «на своём сервере»** — она неверна | Вручную в BotFather |
| Оферта и политика | Опубликовать в Telegraph → `OFFER_URL` / `PRIVACY_URL` в `.env` + Privacy Policy в BotFather | Перезапуск бота |
| Схема БД | `db.py` → `init()`. Только `CREATE TABLE IF NOT EXISTS` / `ALTER TABLE ADD COLUMN`; **перед изменением сделать бэкап `bot.db`** | Перезапуск бота |

### Как выкатывать

- **Воркер:** commit → push в `main` → RunPod сам пересобирает (до 30 мин). Статусы «rolled back / Old» и «rollout 0/1» — нормальная промежуточная картина: новый релиз включается, когда воркер поднимается на первом запросе. Если меняли типы GPU, старые FlashBoot-воркеры нужно убить вручную (или переключить Max workers 0 → 1).
- **Бот:** заменить файлы в `C:\Projects\Transcipt-main\tg-bot` → `Restart-Service PoslovnicaBot` → проверить `tg-bot\logs\bot.log`.
  **Никогда не запускать `python bot.py` руками**: служба уже держит polling, будет `TelegramConflictError`.
- **Проверка воркера** — тест в консоли RunPod:
  ```json
  {"input":{"media_url":"https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac","options":{"enableDiarization":true,"speakerCount":1},"file_name":"jfk.flac","file_type":"audio"}}
  ```
  Ожидаемо: `COMPLETED`, в ответе есть `segments`, `speakers`, `summary`.

---

## 6. Эксплуатация VPS (Windows)

- Службы NSSM: `TelegramBotAPI` (стартует первой) и `PoslovnicaBot` (зависит от неё: `sc.exe config PoslovnicaBot depend= TelegramBotAPI`).
- **Чужие службы `VoinzDashboardWeb` и `AlkorOS` не трогать.**
- Кириллица в логах: `nssm set PoslovnicaBot AppEnvironmentExtra PYTHONIOENCODING=utf-8`.
- Особенности Windows: команда — `python`, а не `python3`; PowerShell разрезает `-DX=3.5` (передавать через `$env:`); переносы строк обратной кавычкой ломаются при вставке — давать однострочные команды.
- Планировщик коверкает инлайн-команды в аргументах задачи — только вызов `.ps1`-файла.
- Бэкапы: раз в неделю копировать `tg-bot\backups` за пределы VPS (вручную).

---

## 7. Известные подводные камни

| Симптом | Причина | Что делать |
|---|---|---|
| `NameError: torch` в whisperx | transformers новее 4.46.3 требует torch ≥ 2.5 | Держать пин `transformers==4.46.3` |
| `libcudnn_ops_infer.so.8` не найден | CTranslate2 нужен cuDNN 8 | Слой `/opt/cudnn8` + `LD_LIBRARY_PATH` в Dockerfile — не удалять |
| `CUDA error: no kernel image` | Задача попала на L4, Blackwell или Ada | Оставить только 3090/A5000, убить старые воркеры |
| pyannote падает при импорте | Нет `matplotlib` | Он уже в requirements |
| Нет спикеров в ответе | Не задан `HF_TOKEN` или pyannote упал (в логе `[diarize] пропущено`) | Проверить env и логи воркера |
| `[Errno 2] ... C:\telegram-bot-api\data\...` | Повторный `file_id`: локальный Bot API отдаёт путь к уже удалённому файлу | Обрабатывается `FileGoneError`. Правильное решение — дедупликация по `file_id` в `jobs` (не сделано) |
| Токен бота в тексте ошибки | Путь содержит токен (`:` заменено на `~`) | Все ошибки пропускать через `_safe_err()` |
| `TelegramConflictError` | Второй экземпляр бота | Остановить ручной запуск, работать только через службу |
| YouTube не скачивается | Блокировка IP дата-центров | `YTDLP_COOKIES`; основной упор — VK, RuTube, Дзен |
| Двойное начисление | — | Защищено `payments.charge_id UNIQUE` |
| Суммы в статистике не сходятся | У Stars `total_amount` в штуках, у ЮKassa — в копейках | Фильтровать по валюте |
| Долгий холодный старт | Модели качаются при старте (§0 п. 3) | Вшить Whisper и выравнивание в образ или использовать RunPod Cached models |

---

## 8. Открытые задачи (по приоритету)

Статус на 18.08 и 16.09. **Перед работой сверить с владельцем.**

**Технические**
1. **Копия бота с VPS опубликована** в `codex/sync-vps-2026-10-27`, Draft PR #1. Исправления и локальные проверки описаны в §0.2. Перед выпуском нужен тестовый прогон внешних интеграций; push в main запускает сборку воркера.
2. **Потоковая загрузка уже реализована в копии VPS:** `fetch_to_disk` → `storage.upload_file(path)` через `asyncio.to_thread` → очистка сразу после загрузки и в `finally`.
3. 🔴 Дедупликация по `file_id`: хранить его в `jobs` и при повторной отправке отдавать готовую расшифровку.
4. 🟡 Проверить execution timeout RunPod (нужно ≥ 3600 с); клиентский таймаут уже 3600 с.
5. 🟡 Холодный старт: вшить Whisper и выравнивание (~4,5 ГБ) или включить Cached model.
6. 🟡 Годовые тарифы (детали — в `tg-bot-status.md`, раздел «Осталось»).
7. 🟢 Обновить torch → появятся 4090/5090/L4.
8. **Счётчик «нас уже N» (`SOCIAL_PROOF_MIN`) реализован** в полученной копии VPS.
9. 🟢 Проверить диаризацию на реальной записи с несколькими спикерами.
10. 🟢 Правки промптов из внешнего разбора — только после `/evaltask`. Список признанных и отклонённых замечаний — в `tg-bot-status.md`. Отклонённое для 3B-модели не возвращать.

**Юридические и денежные**
- Чеки НПД: ЮKassa не пробивает их для самозанятых. Сейчас чек делается вручную в «Мой налог» в день оплаты, затем `/receipt`. Варианты автоматизации: lknpd API, Робокасса «Робочеки СМЗ» или переход на ИП.
- Правила Telegram требуют оплату цифровых товаров в Stars → ЮKassa в боте их нарушает. План: Stars в боте + внешняя страница оплаты картой и СБП.
- Опубликовать оферту и политику; убрать «свой сервер» из описания и с баннера; консультация по 152-ФЗ.

**Продуктовые (Этап 0 стратегии):** события воронки и `/stats` по когортам; себестоимость ₽/час по счетам RunPod; корпус из 20 записей для замера WER/DER; лендинг с Метрикой; кастдев. Роадмап по кварталам — в `claude/strategy-2026-2027.md`.

---

## 9. Документы проекта и что в каждом

| Документ | Что внутри | Когда читать |
|---|---|---|
| `AGENT-HANDOFF.md` | Этот файл; копия в рабочем Git-репозитории и папке Transcribe | Первым |
| `tg-bot-status.md` | Проверенное локально состояние после синхронизации 27.10.2026 | Перед изменениями бота |
| `claude/project-summary.md` | Полная хронология, этапы 1–8, таблица проблем | Нужен контекст «почему так» |
| `claude/tg-bot-status.md` | Бот, платежи, Bot API, Errno 2, промпты, годовые тарифы, `.env`, бэкапы | Любая работа с ботом |
| `claude/code-review-and-local-stack.md` | Gemini-версия, деплой RunPod, фиксы зависимостей | Работа с воркером и веб-версией |
| `claude/tg-bot-branding.md` | BotFather: тексты, команды, чек-лист, палитра | Оформление |
| `claude/strategy-2026-2027.md` | Рынок, сегменты, цены, роадмап | Продукт и приоритеты |
| `claude/case-study-investors.md` | Кейс для инвесторов AMANA | Презентации |

---

## 10. Правила работы для агента

1. Не коммить секреты, `.env`, `bot.db`, логи и бэкапы — это уже прописано в `.gitignore`.
2. Push в `main` = продакшен-деплой воркера. Для экспериментов — отдельная ветка и отдельный эндпоинт RunPod.
3. Перед изменением схемы БД — бэкап `bot.db`.
4. Промпты: сначала замер (`/evaltask`), потом правки. Модель 3B: короткие и конкретные инструкции, без длинных общих system prompt.
5. Команды для владельца давай на PowerShell, одной строкой, с полными путями.
6. Не обещай «обработку на своём сервере»: файлы обрабатываются на RunPod и хранятся в R2.
7. После значимых изменений обнови этот файл и `tg-bot-status.md` в проекте с датой.
