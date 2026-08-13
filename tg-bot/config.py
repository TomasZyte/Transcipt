"""Конфигурация бота из переменных окружения (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# RunPod Serverless эндпоинт (тот, что уже работает)
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_BASE = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# S3-совместимое хранилище (R2 / B2 / MinIO) — файлы для воркера
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_BUCKET = os.getenv("S3_BUCKET", "transcribe")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
S3_REGION = os.getenv("S3_REGION", "auto")
PRESIGN_TTL = int(os.getenv("PRESIGN_TTL", "3600"))

DB_PATH = os.getenv("DB_PATH", "bot.db")

# Админы (Telegram user id через запятую) — доступ к генерации промокодов
ADMINS = {int(x) for x in os.getenv("ADMINS", "").replace(" ", "").split(",") if x}

# Тарифы: план -> лимит минут в месяц (None = безлимит), макс. длина файла (мин)
PLANS = {
    "free":  {"title": "Бесплатный", "minutes": 90,   "max_file_min": 30},
    "start": {"title": "Старт",      "minutes": 1200, "max_file_min": 120},
    "pro":   {"title": "Про",        "minutes": 3600, "max_file_min": 240},
}

# Языки перевода: код -> подпись
LANGS = {
    "ru": "Русский", "kk": "Қазақша", "en": "English", "de": "Deutsch",
    "es": "Español", "zh": "中文", "fr": "Français",
}
