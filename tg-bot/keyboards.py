"""Клавиатуры бота."""
from aiogram.types import (ReplyKeyboardMarkup, KeyboardButton,
                           InlineKeyboardMarkup, InlineKeyboardButton)
import config

main_menu = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🎙 Новая транскрибация")],
        [KeyboardButton(text="Тарифы"), KeyboardButton(text="Узнать больше"),
         KeyboardButton(text="Настройки")],
    ],
    resize_keyboard=True,
)


def result_kb(job_id: int) -> InlineKeyboardMarkup:
    j = str(job_id)
    b = InlineKeyboardButton
    return InlineKeyboardMarkup(inline_keyboard=[
        [b(text="📄 Краткое содержание", callback_data=f"t:{j}:summary"),
         b(text="🔑 Тезисы", callback_data=f"t:{j}:highlights")],
        [b(text="📋 Протокол встречи", callback_data=f"t:{j}:protocol"),
         b(text="✅ Задачи", callback_data=f"t:{j}:tasks")],
        [b(text="🔢 Цифры/даты/бюджет", callback_data=f"t:{j}:numbers"),
         b(text="💬 Спросить AI", callback_data=f"t:{j}:qa")],
        [b(text="🌍 Перевод", callback_data=f"tr:{j}:menu"),
         b(text="⬇️ Скачать", callback_data=f"ex:{j}:menu")],
        [b(text="🏠 Главное меню", callback_data="home")],
    ])


def buy_kb() -> InlineKeyboardMarkup:
    """Рубли — основной способ, звёзды — второй.
    Правила Telegram требуют возможности оплатить цифровые услуги в Stars."""
    b = InlineKeyboardButton
    rows = []

    if config.PAYMENTS_ENABLED:
        rows.append([b(text=f"💳 Старт — {config.PRICES['start'] // 100} ₽",
                       callback_data="buy:start")])
        rows.append([b(text=f"💳 Про — {config.PRICES['pro'] // 100} ₽",
                       callback_data="buy:pro")])

    if config.STARS_ENABLED:
        rows.append([b(text=f"⭐ Старт — {config.PRICES_XTR['start']}",
                       callback_data="stars:start"),
                     b(text=f"⭐ Про — {config.PRICES_XTR['pro']}",
                       callback_data="stars:pro")])

    return InlineKeyboardMarkup(inline_keyboard=rows)


def export_kb(job_id: int) -> InlineKeyboardMarkup:
    j = str(job_id)
    b = InlineKeyboardButton
    return InlineKeyboardMarkup(inline_keyboard=[[
        b(text=".srt", callback_data=f"ex:{j}:srt"),
        b(text=".txt", callback_data=f"ex:{j}:txt"),
        b(text=".md", callback_data=f"ex:{j}:md"),
    ]])


def translate_kb(job_id: int) -> InlineKeyboardMarkup:
    j = str(job_id)
    rows, row = [], []
    for code, title in config.LANGS.items():
        row.append(InlineKeyboardButton(text=title, callback_data=f"tr:{j}:{code}"))
        if len(row) == 2:
            rows.append(row); row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(inline_keyboard=rows)