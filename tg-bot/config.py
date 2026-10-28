"""Конфигурация бота из переменных окружения (.env)."""
import os
from dotenv import load_dotenv

# The dedicated test entry point selects an explicit file before importing the bot.
# Do not merge inherited production values or implicitly discover another .env.
TEST_SETTINGS = None
if "POSLOVNICA_TEST_CONFIG" in os.environ:
    from pathlib import Path
    from integration_config import TestSettings
    TEST_SETTINGS = TestSettings.load(Path(os.environ["POSLOVNICA_TEST_CONFIG"]))
    os.environ.update(TEST_SETTINGS.environment())
else:
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

# ЮKassa через Telegram Payments (@BotFather → Payments → ЮKassa)
PROVIDER_TOKEN = os.getenv("PROVIDER_TOKEN", "").strip()
PAYMENTS_ENABLED = bool(PROVIDER_TOKEN)
IS_TEST_PAYMENTS = ":TEST:" in PROVIDER_TOKEN

# Цена за 1 месяц, в копейках (RUB). В send_invoice уходит как есть, без *100.
PRICES = {"start": 59000, "pro": 129000}

# Срок, на который начисляется подписка после оплаты
SUB_DAYS = 30

# Валюта счёта
CURRENCY = "RUB"

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

OFFER_URL = "https://telegra.ph/Publichnaya-oferta-servisa-Poslovnica-08-15"
PRIVACY_URL = "https://telegra.ph/Politika-konfidencialnosti-servisa-Poslovnica-08-15"

# --- Telegram Stars: второй способ оплаты ---
STARS_ENABLED = os.getenv("STARS_ENABLED", "1") == "1"
# Цена в звёздах. Ориентир: разработчик получает ~$0,013 за звезду при выводе,
# пользователь покупает их примерно вдвое дороже. 500 ⭐ ≈ 590 ₽ вам на руки.
# Курс меняется — сверьтесь перед запуском и правьте эти числа.
PRICES_XTR = {"start": 500, "pro": 1100}

# --- Публичный счётчик «нас уже N» ---
SOCIAL_PROOF_MIN = int(os.getenv("SOCIAL_PROOF_MIN", "500"))

# --- Свой Telegram Bot API сервер ---
# Пусто = облако Telegram с лимитом 20 МБ. Заполните после установки сервера.
LOCAL_BOT_API = os.getenv("LOCAL_BOT_API", "").strip()
TG_TIMEOUT = int(os.getenv("TG_TIMEOUT", "1800"))
