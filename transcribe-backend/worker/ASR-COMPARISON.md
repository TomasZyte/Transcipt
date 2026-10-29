# Изолированное сравнение ASR

`asr_comparison.py` — отдельный CLI для одного локального WAV и одного варианта.
По умолчанию он только проверяет вход: без ML-импортов, скачивания весов и
инференса. Рабочий pipeline и действующие Dockerfile его не используют.
Он не читает секреты и не отправляет запросы в RunPod, R2 или Telegram.

| ID | Модель и путь | Настройка |
|---|---|---|
| `wx_auto` | WhisperX 3.8.6, large-v3 | auto, VAD 0.500/0.363, batch=8 |
| `wx_ru` | То же | Только язык ru |
| `wx_ru_vad` | То же | ru, VAD 0.300/0.200 |
| `fw_v3` | faster-whisper 1.2.1, large-v3 | ru, без VAD, temperature=0, beam=5 |
| `fw_v2` | faster-whisper 1.2.1, large-v2 | Только другие веса относительно fw_v3 |
| `gigaam_v3` | GigaAM v3_rnnt | Отдельное окружение, короткий transcribe, FP16 encoder |

Параметры заданы в коде; произвольные prompts/hotwords не принимаются. Сравнение
прямого faster-whisper с WhisperX меняет весь путь распознавания, а не только VAD.

## Подготовка без модели

В `audio-manifest.json` рядом с WAV нужен список `clips`: `id`, `file` (только имя),
`sha256`, `source_start_seconds`, `source_end_seconds`, опционально `duration_seconds`.
Проверяются PCM, хеш, размер до 2 МиБ и временные смещения. Формат — mono/16kHz/PCM16;
максимум 60 с для Whisper и 25 с для короткого GigaAM. Эталон не читается.

Из корня репозитория, одна строка PowerShell:

```powershell
.\.venv\Scripts\python.exe transcribe-backend\worker\asr_comparison.py --manifest .test-runtime\quality-2026-10-28\asr-children-comparison\audio-manifest.json --clip context --variant wx_auto --output .test-runtime\quality-2026-10-28\asr-children-comparison\wx_auto-plan.json
```

Результат: `status=not_run`; текст, стоимость, WER/DER — `null`. Выходной файл
должен быть новым. Основной вход sound2 — 8.5–19.5 с; дополнительные окна —
проверка влияния контекста, не независимые записи.

## Явное выполнение в будущем

Одновременно нужны `--execute`, `ASR_COMPARISON_TEST=1` и `--model-lock`.
Один процесс обрабатывает один вариант, что исключает перенос настроек между
кэшированными моделями. GigaAM требует отдельного окружения.

Пример структуры lock (заглушки намеренно непригодны для запуска):

```json
{
  "model": "large-v3",
  "revision": "<настоящая ревизия, 40–64 hex>",
  "root": "models/large-v3",
  "files": {
    "model.bin": "<sha256>",
    "config.json": "<sha256>",
    "tokenizer.json": "<sha256>"
  },
  "packages": {
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.5.0",
    "torch": "<точная установленная версия>"
  }
}
```

`files` должен перечислять **все** файлы каталога весов, включая словарь и
препроцессор, если они есть. Каждый хеш проверяется до загрузки модели. Лишние
файлы и ссылки запрещены. Версии `packages` сверяются с установленными.
Для WhisperX дополнительно обязательны `whisperx=3.8.6`, `pyannote.audio`,
`transformers`, `torchaudio` и `vad: {path, sha256}` для локальной модели VAD.
Остальные версии брать из проверенного test-stack, не менять production.

Для GigaAM: `model=v3_rnnt`, файл `v3_rnnt.ckpt`, пакеты `torch`, `gigaam`,
`omegaconf`, `hydra-core`. `revision` — полный commit официального репозитория;
он сверяется с VCS-метаданными установленного пакета. Загрузчик получает имя
модели и проверенный каталог кэша. Передавать штатный checkpoint как путь нельзя:
upstream использует такой вход для другой схемы fine-tuned checkpoint.

Веса и окружения собраны и проверены на временных CPU-runner GitHub Actions,
см. результат ниже. Локально веса не скачивались. Model-lock создан внутри
каждого тестового образа. GPU-совместимость и качество ASR ещё не проверены.

До модели создаётся эксклюзивный `*.started.json`. Повтор с тем же выходом
запрещён, в том числе после ошибки. `*.failed.json` содержит только тип ошибки;
автоматических повторов нет. HF offline-переменные и запрет Python socket
защищают от случайной загрузки, но не заменяют сетевую песочницу ОС.

CLI **не управляет облачным бюджетом и timeout GPU**. Для будущего RunPod-прогона
нужны отдельное разрешение, сверенный лимит и execution timeout. Текущий
`quality_batch_handler.py` этот CLI не вызывает; отдельная точка входа описана ниже.
Код опубликован только в отдельной ветке CPU-сборки `codex/asr-build-check`.
Endpoint и ветки, подключённые к RunPod, не изменены.

## Подготовленные тестовые образы — 29.10.2026

Добавлены `Dockerfile.asr-whisper` и `Dockerfile.asr-gigaam`. Они не заменяют
production Dockerfile или действующий `Dockerfile.quality`. Первый по умолчанию
содержит large-v3 и позволяет `wx_auto`, `wx_ru`, `wx_ru_vad`, `fw_v3`. Для отдельного
контроля large-v2 используется build-arg `ASR_MODEL=large-v2` и вариант `fw_v2`.
Второй образ содержит только GigaAM и отдельные зависимости.

`asr_model_sources.json` фиксирует:

- large-v3: `edaa852ec7e145841d8ffdb056a99866b5f0a478`;
- large-v2: `f0fe81560cb8b68660e564f55dd99207059c092e`;
- код GigaAM: `7447938d791c4f3e643386ee22c33777004293a5`.

Whisper загружается по неизменяемым URL ревизии. Проверяются опубликованные LFS
SHA256 весов, Git blob-хеши остальных файлов и точные размеры. Для GigaAM источник
публикует MD5 checkpoint; после его проверки SHA256 записывается в model-lock.
MD5 — проверка соответствия upstream-файлу, не современная криптографическая
гарантия происхождения. Адреса/хеши сверены с публичными метаданными и кодом.

`prepare_asr_assets.py` без `--download` выводит только план. В Dockerfile явный
`--download` скачивает веса **при сборке**, создаёт `/opt/asr/model-lock.json`,
фиксирует все установленные версии и проверяет их без загрузки модели в GPU.
Whisper-веса занимают около 3,09 ГБ на один вариант; два веса сразу не включаются.
Транзитивные версии фиксируются фактически после установки в образ; полный
воспроизводимый dependency lock до сборки ещё не получен. Базовый образ задан тегом;
перед запуском нужно сохранить digest успешно собранного образа.

Отдельный `asr_comparison_handler.py` принимает:

```json
{
  "profile": "asr-comparison",
  "variant": "wx_auto",
  "sample": {
    "id": "context",
    "sha256": "<sha256 WAV из приватного манифеста>",
    "media_url": "<временный URL только poslovnica-test/quality/>",
    "source_start_seconds": 8.5,
    "source_end_seconds": 19.5,
    "duration_seconds": 11
  }
}
```

Дополнительные поля и пакеты файлов запрещены. До ML-процесса проверяются тестовый
режим, модель образа, R2 host/bucket, WAV и SHA. Временный URL в дочерний процесс
не передаётся. Дочерний CLI работает максимум 100 с; при ошибке/таймауте повтора нет.
Ограничитель Python-сокетов остаётся в дочернем процессе и не мешает SDK RunPod
вернуть ответ. Базовые env: `ASR_COMPARISON_IMAGE=1`, `ASR_IMAGE_MODEL` из образа,
`ASR_COMPARISON_TEST=0` по умолчанию. Для разрешённого теста отдельно задаются
`ASR_COMPARISON_TEST=1` и `QUALITY_R2_HOST` тестового аккаунта.

Сетевые загрузки аудио ограничены 30-секундным socket timeout; дополнительно
нужен жёсткий provider execution timeout, например 180 с, TTL 240 с, один worker
и внешний журнал/отмена клиента. Эти настройки сейчас не применялись. Handler
не гарантирует дедупликацию между перезапусками worker: клиент обязан сохранить
job ID и исключать повторы. Общий бюджет контролируется клиентом, не handler.

Docker CLI локально не найден; оба образа успешно собраны в GitHub Actions.
В каждом прошли pip check, импорт библиотек и проверка model-lock при отключённой
сети. До сборки в каждом задании прошли **78 офлайн-тестов**. Старый бюджет и
настройки RunPod не менялись. GPU-прогон ещё не выполнялся.

### Проверка сборки в GitHub Actions

`.github/workflows/asr-build-check.yml` запускается только при push в отдельную
ветку `codex/asr-build-check` публичного репозитория. Два стандартных Ubuntu CPU
runner независимо собирают Whisper/GigaAM, запускают офлайн-тесты, pip check,
импорт и проверку встроенного model-lock в контейнере без сети. ASR не запускается.
Workflow не получает секретов RunPod/R2/HF, не публикует образы/кэш/артефакты,
не развёртывает endpoint и не запускается в `main` или `codex/transcription-quality`.
Собранный образ существует только на временном runner; для RunPod впоследствии
потребуется отдельная сборка либо явно разрешённая публикация в registry.
Стандартные runner публичных репозиториев бесплатны по
[правилам GitHub](https://docs.github.com/en/billing/concepts/product-billing/github-actions).

Результат 29.10.2026, 12:12 МСК: [обе сборки успешны](https://github.com/TomasZyte/Transcipt/actions/runs/40847234871)
на коммите `93767e8b5beb1c303dc4dfb908db8ae6be9b05f8`.

| Образ | Локальный image ID на runner | Размер без сжатия |
|---|---|---|
| Whisper large-v3 | `sha256:40fa595ae1063404fcde6fcc53d69d7a4d024f6b53a245922e9e42ca5145555e` | 12 343 388 104 байт |
| GigaAM v3_rnnt | `sha256:1730c3ea1e2fd147c7189f9fc36bff32352e4e813a399126a5cbcfd432dcd7c9` | 8 929 114 595 байт |

Это идентификаторы временных локальных образов CI, не адреса для скачивания
из registry. Образы не публиковались и удаляются вместе с runner. Для RunPod
нужна сборка из этой ветки с явным отдельным Dockerfile либо отдельно согласованная
публикация образов. large-v2 в этом запуске не собирался. Аудио владельца,
секреты, endpoint и GPU в workflow не использовались.

## Результат и оценка

Сохраняются сырой ASR, настройки, версии и SHA lock, время загрузки и инференса.
Время загрузки модели не равно полному cold start контейнера или счёту RunPod.
Тайм-коды дополнены смещением в исходный файл. У WhisperX это границы объединённых
VAD-чанков; сырая дорожка отдельно не сохраняется (`vad_spans=null`). У короткого
GigaAM времена остаются `null`; фактические типы параметров модели записываются.
Alignment, диаризация, LLM и исправление текста не выполняются.

Две частичные реплики владельца используются только после инференса. По ним
нельзя считать полный WER/DER. Русский уже определился верно в сохранённом
полнофайловом sound2; если `wx_auto` снова даёт ru, отдельный `wx_ru` избыточен.

Проверка: `python -m unittest discover -s transcribe-backend/worker/tests -v`.
ML-вызовы подменены: тесты проверяют контракты трёх библиотек, генератор,
аудио/хеши/лимиты, отсутствие подсказок, смещения и запрет повторов. Это не
сравнительный прогон моделей. 29.10.2026 прошли 64 офлайн-теста, из них 18 новых.
Первый прогон заблокирован правами временной папки Windows sandbox; повтор вне
песочницы успешен.

API: [WhisperX 3.8.6](https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/asr.py),
[локальный VAD](https://github.com/m-bain/whisperX/blob/v3.8.6/whisperx/vads/pyannote.py),
[faster-whisper 1.2.1](https://github.com/SYSTRAN/faster-whisper/blob/v1.2.1/faster_whisper/transcribe.py),
[GigaAM](https://github.com/salute-developers/GigaAM/blob/main/gigaam/__init__.py).
