# Деплой воркера на RunPod Serverless (сборка Docker на своём ПК)

Итог: свой endpoint, который принимает ссылку на медиафайл и возвращает транскрипт +
диаризацию (+ саммари, когда подключите LLM). Платите только за секунды работы GPU.

## 0. Что нужно заранее
- **Docker** установлен (у вас есть).
- Аккаунт на **Docker Hub** (бесплатный) — https://hub.docker.com → запомните username.
- **Токен HuggingFace** — https://hf.co/settings/tokens (Read). На страницах
  `pyannote/speaker-diarization-3.1` и `pyannote/segmentation-3.0` нажмите **Agree**
  (без этого диаризация не скачается).
- На RunPod **загружены кредиты** (Load credits, ~$10 хватит надолго).

## 1. Собрать образ (в папке `worker/`)
```bash
cd transcribe-backend/worker
docker build -t <DOCKERHUB_USER>/transcribe-worker:latest .
```
Образ большой (~6–8 ГБ) и первая сборка идёт 10–20 мин — это нормально.

## 2. Запушить в Docker Hub
```bash
docker login
docker push <DOCKERHUB_USER>/transcribe-worker:latest
```

## 3. Создать Serverless Endpoint на RunPod
1. Левое меню → **Serverless** → **New Endpoint**.
2. **Docker image**: `<DOCKERHUB_USER>/transcribe-worker:latest`.
3. **GPU**: 24 GB (напр. RTX 4090 / L40S). Можно 16 GB, если поставите `COMPUTE_TYPE=int8`.
4. **Workers**: Min = 0 (scale-to-zero, экономно), Max = 1 (потом поднимете).
5. **Container disk**: ~20 GB (под веса моделей).
6. **Environment variables** (вкладка Env):
   ```
   HF_TOKEN      = hf_xxx                 # обязательно для диаризации
   DEVICE        = cuda
   COMPUTE_TYPE  = float16                # или int8 для 16 ГБ
   ASR_MODEL     = large-v3
   # LLM для саммари — см. раздел 6. Пока можно не задавать: саммари подставится заглушкой,
   # транскрипт и диаризация будут работать.
   ```
7. Deploy.

## 4. Проверить прямо в RunPod
На странице endpoint → вкладка **Requests** → отправьте тестовый JSON
(`media_url` — прямая ссылка на любой публичный mp3/mp4):
```json
{ "input": {
    "media_url": "https://example.com/test.mp3",
    "options": { "enableDiarization": true, "speakerCount": 2 },
    "file_name": "test.mp3", "file_type": "audio"
} }
```
Первый запуск = холодный старт + скачивание моделей (может занять 1–3 мин), дальше быстро.
В ответе должны прийти `segments` со спикерами и `rawText`.

## 5. Подключить к API-серверу
Скопируйте на странице endpoint его **ID** и создайте **RunPod API key**
(Settings → API Keys). На API-сервере (папка `api/`) задайте:
```
DISPATCH_BASE_URL = https://api.runpod.ai/v2/<ENDPOINT_ID>
DISPATCH_API_KEY  = <RUNPOD_API_KEY>
S3_* , REDIS_URL   # хранилище + очередь (см. .env.example)
```
Всё — API кладёт файл в S3, шлёт ссылку на endpoint, опрашивает статус. Фронтенд
переключается на очередь патчем из `README.md`.

## 6. Саммари (LLM) — когда дойдут руки
Воркер запрашивает саммари у OpenAI-совместимого LLM по `LLM_BASE_URL`. На serverless
такого сервиса рядом нет, поэтому есть варианты:
- **Пока не подключать** — транскрипт+диаризация работают, саммари = заглушка (код не падает).
- **Отдельный RunPod-под с Ollama/vLLM** — поднять `qwen2.5:7b-instruct`, указать его URL в `LLM_BASE_URL`.
- **Второй serverless endpoint** только под LLM (масштабируется отдельно).

## Если сборка/запуск ругается
- **WhisperX тянет несовместимый torch** — при сборке видно в логах. Обычно лечится тем,
  что база уже с torch 2.4.1; если конфликт — напишите мне лог, подправим пин версий.
- **Диаризация пустая / 401** — не принят HF-лицензионный доступ к pyannote или нет `HF_TOKEN`.
- **CUDA out of memory** — поставьте `COMPUTE_TYPE=int8` или GPU побольше.
- **Долгий первый запрос** — это загрузка весов; чтобы ускорить, раскомментируйте
  «прогрев весов» в `Dockerfile` и пересоберите.
