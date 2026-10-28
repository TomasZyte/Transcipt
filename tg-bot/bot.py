"""Точка входа Telegram-бота транскрибации (aiogram 3)."""
import asyncio
import html
import logging
import os
import re
import shutil
import time
import traceback
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from functools import wraps

from aiogram import Bot, Dispatcher, F, BaseMiddleware
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.filters import CommandStart, Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (Message, CallbackQuery, BufferedInputFile, FSInputFile,
                           LabeledPrice, PreCheckoutQuery, BotCommand,
                           BotCommandScopeDefault, BotCommandScopeChat, ErrorEvent)

import config
import db
import keyboards as kb
import runpod_client as rp
import storage
import exporters

logging.basicConfig(level=logging.INFO)
dp = Dispatcher(storage=MemoryStorage())

MAINT_KEY = "maintenance"
SUPPORT = "@ak_kardano"          # ← подставь свой контакт поддержки
ASSETS = Path(__file__).parent / "assets"

MENU_TEXTS = {"🎙 Новая транскрибация", "Тарифы", "Узнать больше", "Настройки"}
_photo_cache: dict[str, str] = {}
LINK_RE = re.compile(r"https?://\S+")


# ---------- middleware ----------
class MaintenanceMiddleware(BaseMiddleware):
    """Техработы: все, кроме админов, получают баннер. Оплаты пропускаем всегда."""

    async def __call__(self, handler, event, data):
        user = data.get("event_from_user")
        if user and user.id in config.ADMINS:
            return await handler(event, data)
        if isinstance(event, Message) and event.successful_payment:
            return await handler(event, data)

        text = await db.get_setting(MAINT_KEY)
        if not text:
            return await handler(event, data)

        if isinstance(event, Message):
            await event.answer(f"🛠 {text}")
        elif isinstance(event, CallbackQuery):
            await event.answer(f"🛠 {text}", show_alert=True)
        return None


class TrackMiddleware(BaseMiddleware):
    """Профиль и счётчики пользователя обновляются на каждом действии."""

    async def __call__(self, handler, event, data):
        u = data.get("event_from_user")
        if u:
            is_file = isinstance(event, Message) and bool(
                event.voice or event.audio or event.video or event.video_note or event.document
            )
            source = None
            if isinstance(event, Message) and (event.text or "").startswith("/start "):
                source = event.text.split(maxsplit=1)[1][:64]
            try:
                await db.touch_user(
                    u.id, username=u.username, first_name=u.first_name,
                    last_name=u.last_name, lang_code=u.language_code,
                    is_premium=bool(getattr(u, "is_premium", False)),
                    source=source, is_file=is_file,
                )
            except Exception:
                logging.exception("touch_user failed")
        return await handler(event, data)


dp.message.outer_middleware(MaintenanceMiddleware())
dp.callback_query.outer_middleware(MaintenanceMiddleware())
dp.message.middleware(TrackMiddleware())
dp.callback_query.middleware(TrackMiddleware())


class Flow(StatesGroup):
    qa = State()
    promo = State()
    glossary = State()


# ---------- утилиты ----------
async def send_long(msg: Message, text: str, reply_markup=None):
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)] or [""]
    for idx, ch in enumerate(chunks):
        await msg.answer(ch, reply_markup=reply_markup if idx == len(chunks) - 1 else None)


async def send_banner(m: Message, name: str, caption: str, reply_markup=None):
    """Баннер с подписью. Первый раз грузит файл, дальше — по file_id.
    Нет картинки или Telegram её отверг — молча уходим в обычный текст."""
    try:
        if name in _photo_cache:
            return await m.answer_photo(_photo_cache[name], caption=caption,
                                        parse_mode="HTML", reply_markup=reply_markup)
        path = ASSETS / f"{name}.png"
        if not path.exists():
            return await m.answer(caption, parse_mode="HTML", reply_markup=reply_markup)
        msg = await m.answer_photo(FSInputFile(path), caption=caption,
                                   parse_mode="HTML", reply_markup=reply_markup)
        _photo_cache[name] = msg.photo[-1].file_id
        return msg
    except Exception:
        logging.exception("banner %s failed", name)
        return await m.answer(caption, parse_mode="HTML", reply_markup=reply_markup)

_TOKEN_RE = re.compile(r"\b\d{6,}[:~][A-Za-z0-9_\-]{30,}")


def _safe_err(e: object) -> str:
    """Текст ошибки без токена бота. Локальный Bot API кладёт токен в путь к файлу
    (`...\\data\\<id>~<token>\\music\\file_17.m4a`), и он утекал в лог, в events
    и в админское уведомление."""
    s = str(e)
    tok = getattr(config, "BOT_TOKEN", "") or ""
    if tok:
        s = s.replace(tok, "***").replace(tok.replace(":", "~"), "***")
    return _TOKEN_RE.sub("***", s)


class RedactingFormatter(logging.Formatter):
    def format(self, record):
        # Redact after formatting so exception messages and tracebacks are covered.
        return _safe_err(super().format(record))


for _handler in logging.getLogger().handlers:
    _handler.setFormatter(RedactingFormatter("%(levelname)s:%(name)s:%(message)s"))


class FileGoneError(RuntimeError):
    """getFile вернул путь, которого нет на диске: файл уже обработан и удалён."""


TMP_DIR = Path(__file__).parent / "tmp"


async def fetch_to_disk(bot: Bot, file_id: str) -> Path:
    """Кладёт файл на диск и возвращает путь — в память не читаем.
    При потолке 2 ГБ прежний вариант (весь файл в bytes) укладывал бота по RAM.

    В локальном режиме getFile отдаёт путь на диске: переносим файл к себе.
    На одном томе os.replace мгновенен и не требует места, а рабочая папка
    Bot API не разрастается — сервер сам её не чистит."""
    TMP_DIR.mkdir(exist_ok=True)
    f = await bot.get_file(file_id)
    src = f.file_path
    dst = TMP_DIR / f"{uuid.uuid4().hex}{Path(src or '').suffix or '.bin'}"

    try:
        if getattr(config, "LOCAL_BOT_API", "") and src and os.path.isabs(src):
            if not os.path.exists(src):
                logging.warning("local file missing, file_id=%s", file_id)
                raise FileGoneError(file_id)
            try:
                os.replace(src, dst)  # один том — переименование, без копирования
            except OSError:
                shutil.move(src, dst)  # разные тома — копирование с удалением
        else:
            await bot.download_file(src, dst)
        return dst
    except BaseException:
        if dst.exists():
            _drop_tmp(dst)
        raise


def _drop_tmp(path) -> None:
    try:
        os.remove(path)
    except OSError:
        logging.warning("временный файл не удалён: %s", path)


def _clean_tmp() -> None:
    """Чистка tmp на старте: если бот падал посреди обработки, файлы остались.
    В момент старта ничего не обрабатывается, так что сносим всё."""
    if not TMP_DIR.exists():
        return
    for p in TMP_DIR.iterdir():
        if p.is_file():
            _drop_tmp(p)


_proof_cache = {"at": 0.0, "text": ""}


async def social_proof() -> str:
    """«Нас уже N» — показываем только после порога, иначе цифра работает против нас.
    Результат кешируется на 10 минут, чтобы не считать COUNT на каждый /start."""
    if time.time() - _proof_cache["at"] < 600:
        return _proof_cache["text"]

    text = ""
    try:
        total = await db.count_users()
        if total >= config.SOCIAL_PROOF_MIN:
            s = await db.stats()
            mins = int(s["minutes"])
            nice = lambda n: f"{n:,}".replace(",", " ")
            text = f"\n\n👥 Нас уже {nice(total)}"
            if mins >= 1000:
                text += f", расшифровано {nice(mins)} минут"
            text += "."
    except Exception:
        logging.exception("social_proof failed")

    _proof_cache.update(at=time.time(), text=text)
    return text


TASK_LABELS = {
    "summary":    "Готовлю краткое содержание",
    "highlights": "Выделяю тезисы",
    "protocol":   "Составляю протокол встречи",
    "tasks":      "Ищу задачи и поручения",
    "numbers":    "Собираю цифры и даты",
    "links":      "Собираю упомянутые ресурсы",
}


@asynccontextmanager
async def working(msg: Message, text: str):
    """Статус-сообщение со счётчиком + «печатает…» в шапке чата.
    Индикатор Telegram живёт 5 секунд, поэтому шлём его по кругу."""
    status = await msg.answer(f"⏳ {text}…")
    stop = asyncio.Event()

    async def ticker():
        started = time.time()
        while not stop.is_set():
            try:
                await msg.bot.send_chat_action(msg.chat.id, "typing")
            except Exception:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=5)
            except asyncio.TimeoutError:
                el = int(time.time() - started)
                if el >= 15:
                    try:
                        await status.edit_text(f"⏳ {text}… {el // 60}:{el % 60:02d}")
                    except Exception:
                        pass

    task = asyncio.create_task(ticker())
    try:
        yield status
    finally:
        stop.set()
        try:
            await task
        except Exception:
            pass
        try:
            await status.delete()
        except Exception:
            pass

def is_interrupt(m: Message) -> bool:
    t = (m.text or "").strip()
    return t.startswith("/") or t in MENU_TEXTS


def result_caption(t: dict) -> str:
    s = t.get("summary", {}) or {}
    speakers = len({seg.get("speaker") for seg in t.get("segments", [])})
    dur = int(t.get("durationEstimateSec", 0))
    # Budget in UTF-16 units before escaping; never cut an HTML tag/entity.
    def safe(value, units):
        value = str(value).encode("utf-16-le")[:units * 2].decode("utf-16-le", errors="ignore")
        return html.escape(value)

    head = (f"✅ <b>{safe(t.get('fileName','Запись'), 250)}</b>\n"
            f"Язык: {safe(t.get('languageDetected','?'), 40)} · Спикеров: {speakers} · "
            f"{dur//60} мин {dur%60} сек\n")
    if s.get("overview"):
        head += f"\n<b>Кратко:</b> {safe(s['overview'], 1800)}\n"
    if s.get("keyPoints"):
        head += "\n<b>Тезисы:</b>\n" + "\n".join(f"• {safe(x, 250)}" for x in s["keyPoints"][:6])
    return head


async def balance_text(uid: int) -> str:
    plan = await db.effective_plan(uid)
    left = await db.remaining_minutes(uid)
    u = await db.get_user(uid)
    limit = config.PLANS[plan]["minutes"]

    if plan == "free":
        term = "бесплатный тариф"
    elif not u["plan_until"]:
        term = "бессрочно"
    else:
        term = "действует до " + time.strftime("%d.%m.%Y", time.localtime(u["plan_until"]))

    return (f"<b>Ваш тариф:</b> {config.PLANS[plan]['title']} — {term}\n"
            f"<b>Минуты:</b> {int(left)} из {limit} осталось в этом месяце\n"
            f"<b>Файл за раз:</b> до {config.PLANS[plan]['max_file_min']} мин\n\n"
            f"Лимит минут обнуляется в начале календарного месяца.")


# ---------- команды/меню ----------
@dp.message(CommandStart())
async def start(m: Message, state: FSMContext):
    await state.clear()
    is_new = not await db.user_exists(m.from_user.id)
    await db.get_user(m.from_user.id)
    await db.log_event(m.from_user.id, "start", (m.text or "")[:100])

    free = config.PLANS["free"]
    if is_new:
        caption = (
            "<b>Привет!</b> Я превращаю аудио и видео в текст — кто что сказал, "
            "точные тайм-коды, конспект и задачи. Записи не передаются в публичные "
            "AI-сервисы и не используются для обучения моделей.\n\n"
            f"🎁 <b>{free['minutes']} минут в месяц — бесплатно</b>, прямо сейчас и без "
            f"оплаты. Один файл — до {free['max_file_min']} минут.\n\n"
            "Пришлите голосовое, аудио или видео — обработаю сразу."
            + await social_proof()
        )
    else:
        plan = await db.effective_plan(m.from_user.id)
        left = await db.remaining_minutes(m.from_user.id)
        caption = (
            f"С возвращением! Тариф <b>{config.PLANS[plan]['title']}</b> · "
            f"осталось <b>{int(left)}</b> мин в этом месяце.\n\n"
            "Присылайте файл — расшифрую."
        )

    await send_banner(m, "start", caption, reply_markup=kb.main_menu)

    if is_new:
        u = m.from_user
        await notify_admins(
            m.bot,
            f"👤 <b>Новый пользователь</b>\n"
            f"{html.escape(u.full_name)} (@{u.username or '—'}, <code>{u.id}</code>)\n"
            f"Язык: {u.language_code or '—'}"
            f"{' · Premium' if getattr(u, 'is_premium', False) else ''}\n"
            f"Источник: {html.escape((m.text or '/start').replace('/start', '').strip() or 'прямой заход')}",
        )
        await check_milestone(m.bot)

@dp.message(Command("help"))
async def help_cmd(m: Message, state: FSMContext):
    await state.clear()
    await send_banner(
        m, "help",
        "Пришлите голосовое, аудио или видео — верну текст с разделением по спикерам, "
        "тайм-кодами и конспектом. Дальше по кнопкам: протокол встречи, задачи, "
        "цифры и бюджет, вопрос к AI, перевод, экспорт в TXT/SRT/MD.\n\n"
        "<b>Команды</b>\n"
        "/start — главное меню\n"
        "/balance — тариф и остаток минут\n"
        "/tariffs — тарифы и оплата\n"
        "/settings — промокод и глоссарий\n"
        "/cancel — отменить текущий ввод\n"
        "/terms — оферта и политика\n"
        "/support — поддержка",
        reply_markup=kb.main_menu,
    )


@dp.message(Command("cancel"))
async def cancel_cmd(m: Message, state: FSMContext):
    if await state.get_state() is None:
        return await m.answer("Нечего отменять.", reply_markup=kb.main_menu)
    await state.clear()
    await m.answer("Отменил. Чем помочь?", reply_markup=kb.main_menu)


@dp.message(Command("support"))
async def support_cmd(m: Message, state: FSMContext):
    await state.clear()
    await m.answer(
        f"Поддержка: {SUPPORT}\n\n"
        "Напишите, что случилось, и приложите пример файла — разберёмся. "
        "По оплате, чекам и возвратам — туда же.\n\n"
        "Документы: /terms",
        reply_markup=kb.main_menu,
    )


@dp.message(Command("terms"))
async def terms_cmd(m: Message, state: FSMContext):
    await state.clear()
    await m.answer(
        "<b>Документы сервиса</b>\n\n"
        f"• Публичная оферта: {config.OFFER_URL}\n"
        f"• Политика конфиденциальности: {config.PRIVACY_URL}\n\n"
        "Оплачивая тариф, вы принимаете условия оферты.",
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


@dp.message(Command("balance"))
async def balance_cmd(m: Message, state: FSMContext):
    await state.clear()
    await m.answer(await balance_text(m.from_user.id), parse_mode="HTML",
                   reply_markup=kb.main_menu)


@dp.message(F.text == "🎙 Новая транскрибация")
async def new_tr(m: Message, state: FSMContext):
    await state.clear()
    await m.answer("Пришлите аудио, голосовое или видео файлом или сообщением 👇")


@dp.message(Command("tariffs"))
@dp.message(Command("plans"))
@dp.message(F.text == "Тарифы")
async def tariffs(m: Message, state: FSMContext):
    await state.clear()
    plan = await db.effective_plan(m.from_user.id)
    left = await db.remaining_minutes(m.from_user.id)
    await m.answer(
        f"Ваш тариф: <b>{config.PLANS[plan]['title']}</b> · осталось "
        f"{int(left)} мин в этом месяце\n\n"
        "<b>Бесплатный</b> — 90 мин/мес, разделение по спикерам, конспект, экспорт\n"
        "<b>Старт — 590 ₽/мес</b> — 20 ч/мес, протокол, задачи, цифры, все форматы, файлы до 2 ч\n"
        "<b>Про — 1290 ₽/мес</b> — 60 ч/мес + перевод, Q&A, ссылки, глоссарий, файлы до 4 ч\n"
        "<b>Бизнес</b> — приватный инстанс и интеграции, по договорённости\n\n"
        "Подписка на 30 дней, продление вручную — автосписаний нет.\n"
        "Есть промокод? Настройки → «Ввести промокод».",
        parse_mode="HTML",
        reply_markup=kb.buy_kb() if (config.PAYMENTS_ENABLED or config.STARS_ENABLED) else None,
    )


# ---------- оплата ----------
@dp.callback_query(F.data.startswith("buy:"))
async def buy(c: CallbackQuery, bot: Bot):
    plan = c.data.split(":")[1]
    if plan not in config.PRICES:
        return await c.answer("Неизвестный тариф.", show_alert=True)
    if not config.PAYMENTS_ENABLED:
        return await c.answer("Оплата пока не подключена.", show_alert=True)

    title = config.PLANS[plan]["title"]
    p = config.PLANS[plan]
    await db.log_event(c.from_user.id, "invoice", plan)
    await bot.send_invoice(
        chat_id=c.from_user.id,
        title=f"Тариф «{title}» — 1 месяц",
        description=(f"{p['minutes'] // 60} ч расшифровки в месяц, файлы до "
                     f"{p['max_file_min'] // 60} ч, разделение по спикерам, "
                     f"конспект и инструменты встреч. Доступ на 30 дней."),
        payload=f"plan:{plan}:{c.from_user.id}:{int(time.time())}",
        provider_token=config.PROVIDER_TOKEN,
        currency=config.CURRENCY,
        prices=[LabeledPrice(label=f"Тариф {title}", amount=config.PRICES[plan])],
        start_parameter=f"buy-{plan}",
        need_email=True,               # email нужен для чека НПД
        send_email_to_provider=False,  # ЮKassa за самозанятых чеки не пробивает
    )
    await c.answer()


@dp.callback_query(F.data.startswith("stars:"))
async def buy_stars(c: CallbackQuery, bot: Bot):
    """Оплата звёздами. Правила Telegram требуют, чтобы цифровые услуги можно было
    оплатить в Stars, поэтому держим как второй способ."""
    plan = c.data.split(":")[1]
    if plan not in config.PRICES_XTR:
        return await c.answer("Неизвестный тариф.", show_alert=True)
    if not config.STARS_ENABLED:
        return await c.answer("Оплата звёздами отключена.", show_alert=True)

    title = config.PLANS[plan]["title"]
    p = config.PLANS[plan]
    await db.log_event(c.from_user.id, "invoice_xtr", plan)
    await bot.send_invoice(
        chat_id=c.from_user.id,
        title=f"Тариф «{title}» — 1 месяц",
        description=(f"{p['minutes'] // 60} ч расшифровки в месяц, файлы до "
                     f"{p['max_file_min'] // 60} ч, разделение по спикерам, "
                     f"конспект и инструменты встреч. Доступ на 30 дней."),
        payload=f"plan:{plan}:{c.from_user.id}:{int(time.time())}",
        provider_token="",             # для Stars токен провайдера обязан быть пустым
        currency="XTR",
        prices=[LabeledPrice(label=f"Тариф {title}", amount=config.PRICES_XTR[plan])],
        start_parameter=f"buy-{plan}-xtr",
    )
    await c.answer()


def invoice_plan(payload: str, uid: int, currency: str, amount: int,
                 *, checkout: bool = False) -> str | None:
    try:
        kind, plan, owner, stamp = payload.split(":")
        if kind != "plan" or int(owner) != uid or int(stamp) < 0:
            return None
        if currency == "XTR":
            prices, enabled = config.PRICES_XTR, config.STARS_ENABLED
        elif currency == config.CURRENCY:
            prices, enabled = config.PRICES, config.PAYMENTS_ENABLED
        else:
            return None
        if (not checkout or enabled) and plan in prices and amount == prices[plan]:
            return plan
    except (ValueError, AttributeError):
        pass
    return None


@dp.pre_checkout_query()
async def pre_checkout(q: PreCheckoutQuery, bot: Bot):
    ok = invoice_plan(q.invoice_payload, q.from_user.id, q.currency, q.total_amount,
                      checkout=True) is not None
    err = None if ok else "Счёт устарел — откройте оплату заново."
    await bot.answer_pre_checkout_query(q.id, ok=ok, error_message=err)


@dp.message(F.successful_payment)
async def paid(m: Message, bot: Bot, state: FSMContext):
    await state.clear()
    sp = m.successful_payment
    plan = invoice_plan(sp.invoice_payload, m.from_user.id, sp.currency, sp.total_amount)
    if plan is None:
        # Money has already moved. Do not silently accept a mismatched invoice.
        await db.log_event(m.from_user.id, "payment_review", sp.telegram_payment_charge_id)
        await notify_admins(bot, "⚠️ Платёж требует ручной проверки: "
                            f"<code>{html.escape(sp.telegram_payment_charge_id)}</code>, "
                            f"пользователь {m.from_user.id}, {sp.total_amount} {html.escape(sp.currency)}")
        return await m.answer(f"Платёж получен, но требует проверки. Напишите {SUPPORT}.")
    oi = sp.order_info

    result = await db.apply_payment(
        uid=m.from_user.id, plan=plan, amount=sp.total_amount, currency=sp.currency,
        charge_id=sp.telegram_payment_charge_id,
        provider_charge_id=sp.provider_payment_charge_id,
        email=oi.email if oi else None,
        phone=oi.phone_number if oi else None,
        name=oi.name if oi else None,
        is_test=config.IS_TEST_PAYMENTS and sp.currency != "XTR",
    )
    if result is None:
        return

    pid, until = result
    stars = sp.currency == "XTR"
    amount_txt = (f"{sp.total_amount} ⭐" if stars
                  else f"{sp.total_amount / 100:.0f} {sp.currency}")
    await db.log_event(m.from_user.id, "pay", f"{plan} {amount_txt}")
    title = config.PLANS[plan]["title"]
    until_txt = "бессрочно" if until == 0 else time.strftime("%d.%m.%Y", time.localtime(until))

    await m.answer(
        f"Оплата прошла ✅ Тариф «{title}» активен до {until_txt}.\n"
        + ("Возврат звёзд — через поддержку.\n\n" if stars
           else "Чек пришлю отдельным сообщением в течение дня.\n\n")
        + "Присылайте запись — работаем.",
        reply_markup=kb.main_menu,
    )

    await notify_admins(
        bot,
        f"💰 <b>Оплата #{pid}</b>{' [ТЕСТ]' if config.IS_TEST_PAYMENTS and not stars else ''}\n"
        f"Тариф: {title} · {amount_txt} · до {until_txt}\n"
        f"Юзер: {html.escape(m.from_user.full_name)} (@{m.from_user.username or '—'}, "
        f"id <code>{m.from_user.id}</code>)\n"
        f"Email: {html.escape((oi.email if oi else None) or '—')}\n"
        + (f"Charge id: <code>{sp.telegram_payment_charge_id}</code>\n\n"
           f"Возврат: <code>/refundstars {m.from_user.id} {sp.telegram_payment_charge_id}</code>"
           if stars else
           f"ЮKassa: <code>{sp.provider_payment_charge_id}</code>\n\n"
           f"Пробей чек в «Мой налог» и отправь:\n"
           f"<code>/receipt {pid} ССЫЛКА</code>"),
    )


@dp.message(F.text == "Узнать больше")
async def about(m: Message, state: FSMContext):
    await state.clear()
    await send_banner(
        m, "about",
        "<b>Poslovnica — расшифровка встреч, звонков и интервью</b>\n\n"
        "Загружаете запись — получаете готовый документ: кто что сказал, точные "
        "тайм-коды, конспект и список задач. Результат можно скачать и отредактировать.",
    )
    await m.answer(
        "<b>Что можно прислать</b>\n"
        "Голосовые, аудиофайлы (MP3, WAV, M4A, OGG), видео (MP4, MOV), видеокружки. "
        "Совещания, интервью, созвоны, лекции, диктофонные записи.\n\n"

        "<b>Что получите</b>\n"
        "• Полный текст с разбивкой по спикерам — видно, кто какую реплику сказал\n"
        "• Тайм-коды на каждую фразу: субтитры SRT не плывут и совпадают с записью\n"
        "• Конспект: краткая выжимка и тезисы\n"
        "• Протокол встречи, список задач с ответственными, цифры и суммы\n"
        "• Вопрос к AI по содержанию записи и перевод на 7 языков\n"
        "• Экспорт в TXT, SRT и Markdown\n\n"

        "<b>Как это устроено</b>\n"
        "Распознавание — Whisper large-v3, русский и ещё 90+ языков. Разделение "
        "спикеров — по всей записи целиком, а не по кускам, поэтому один человек не "
        "распадается на трёх. Конспект считает языковая модель на том же сервере.\n\n"

        "<b>Приватность</b>\n"
        "Для обработки файлы загружаются в Cloudflare R2 и передаются на GPU RunPod. "
        "После загрузки локальная временная копия удаляется. Расшифровка сохраняется "
        "в базе бота для экспорта и работы инструментов.\n\n"

        "<b>Кому подходит</b>\n"
        "Руководителям и ассистентам — протоколы совещаний. Продажам — разбор звонков. "
        "Исследователям и журналистам — интервью. Юристам и врачам — диктовка, где "
        "важно, чтобы запись не ушла на сторону.\n\n"

        "<b>Честные ограничения</b>\n"
        "Сильный шум, музыка на фоне и одновременная речь снижают точность. Файл "
        "принимается до 2 ГБ — хватает и на многочасовое видео. Первая расшифровка "
        "после паузы занимает на минуту дольше: сервер поднимается из спящего "
        "режима.\n\n"

        f"Вопросы — {SUPPORT}. Документы — /terms" + await social_proof(),
        parse_mode="HTML",
        reply_markup=kb.main_menu,
    )


@dp.message(Command("settings"))
@dp.message(F.text == "Настройки")
async def settings(m: Message, state: FSMContext):
    await state.clear()
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    await m.answer("Настройки:", reply_markup=InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎁 Ввести промокод", callback_data="set:promo")],
        [InlineKeyboardButton(text="📚 Глоссарий терминов", callback_data="set:glossary")],
    ]))


@dp.message(Command("promo"))
async def promo_cmd(m: Message, state: FSMContext):
    parts = m.text.split(maxsplit=1)
    if len(parts) == 2:
        await state.clear()
        ok, txt = await db.redeem_promo(m.from_user.id, parts[1].strip().upper())
        await m.answer(txt)
    else:
        await state.set_state(Flow.promo)
        await m.answer("Введите промокод (или /cancel):")


# ---------- админские команды ----------
async def notify_admins(bot: Bot, text: str):
    for admin in config.ADMINS:
        try:
            await bot.send_message(admin, text, parse_mode="HTML")
        except Exception:
            logging.exception("admin notify failed for %s", admin)


MILESTONES = [200, 400, 500, 700, 1000, 1500, 2000]   # дальше — каждые +500


def _milestone_hit(total: int) -> int | None:
    """Наибольший достигнутый порог или None."""
    best = None
    for m in MILESTONES:
        if total >= m:
            best = m
    if total >= 2500:
        best = max(best or 0, (total // 500) * 500)
    return best


async def check_milestone(bot: Bot):
    """Уведомляет о взятом пороге ровно один раз — отметка живёт в settings."""
    total = await db.count_users()
    hit = _milestone_hit(total)
    if not hit:
        return
    if hit <= int(await db.get_setting("milestone", "0") or 0):
        return
    await db.set_setting("milestone", str(hit))

    s = await db.stats()
    if hit == 500:
        head = "🚀🚀🚀 <b>ВНИМАНИЕ — 500 ПОЛЬЗОВАТЕЛЕЙ!</b> 🚀🚀🚀"
    elif hit >= 1000:
        head = f"🏆 <b>{hit} пользователей!</b>"
    else:
        head = f"🎉 <b>{hit} пользователей!</b>"

    await notify_admins(
        bot,
        f"{head}\n\n"
        f"Всего в базе: {total}\n"
        f"Платных: {s['paid']} · активных за 7 дней: {s['active_7d']}\n"
        f"Расшифровок: {s['jobs']} · минут {int(s['minutes'])}\n"
        f"Выручка всего: {s['pay_sum'] / 100:.0f} ₽",
    )


@dp.message(Command("newpromo"))
async def newpromo(m: Message):
    if m.from_user.id not in config.ADMINS:
        return
    try:
        _, code, plan, months, uses = m.text.split()
        await db.create_promo(code.upper(), plan, int(months), int(uses))
        await m.answer(f"Промокод {code.upper()} создан: {plan}, {months} мес, активаций {uses}.")
    except ValueError:
        await m.answer("Формат: /newpromo КОД plan months uses\nНапр.: /newpromo VOINZ-ALFA pro 0 -1")


@dp.message(Command("receipt"))
async def receipt_cmd(m: Message, command: CommandObject, bot: Bot):
    if m.from_user.id not in config.ADMINS:
        return
    parts = (command.args or "").split()
    if len(parts) != 2 or not parts[0].isdigit():
        return await m.answer("Формат: /receipt ID_платежа ссылка_на_чек")

    row = await db.set_receipt(int(parts[0]), parts[1])
    if not row:
        return await m.answer("Платёж не найден.")
    try:
        await bot.send_message(
            row["user_id"],
            f"🧾 Чек по оплате тарифа «{config.PLANS[row['plan']]['title']}»:\n{parts[1]}",
        )
        await db.mark_receipt_sent(int(parts[0]), parts[1])
        await m.answer(f"Чек по платежу #{parts[0]} отправлен.")
    except Exception as e:
        await m.answer(f"Чек записан, но отправить не вышло: {_safe_err(e)}")


@dp.message(Command("refundstars"))
async def refund_stars_cmd(m: Message, command: CommandObject, bot: Bot):
    """Возврат оплаты звёздами. Telegram требует от разработчика поддерживать возвраты."""
    if m.from_user.id not in config.ADMINS:
        return
    parts = (command.args or "").split()
    if len(parts) != 2 or not parts[0].isdigit():
        return await m.answer("Формат: /refundstars user_id charge_id")
    try:
        await bot.refund_star_payment(user_id=int(parts[0]),
                                      telegram_payment_charge_id=parts[1])
        await m.answer("Звёзды возвращены.")
        await bot.send_message(int(parts[0]),
                               "Оплата звёздами возвращена. Если это ошибка — напишите "
                               f"{SUPPORT}.")
    except Exception as e:
        await m.answer(f"Не вышло: {e}")


@dp.message(Command("pending"))
async def pending_cmd(m: Message):
    if m.from_user.id not in config.ADMINS:
        return
    rows = await db.pending_receipts()
    if not rows:
        return await m.answer("Все чеки отправлены 👍")
    await m.answer("Ждут чека:\n" + "\n".join(
        f"#{r['id']} · "
        + (f"{r['amount']} ⭐" if r['currency'] == 'XTR' else f"{r['amount'] / 100:.0f} ₽")
        + f" · {config.PLANS[r['plan']]['title']} · "
        f"{r['email'] or 'без email'} · "
        f"{time.strftime('%d.%m', time.localtime(r['created']))}" for r in rows))


@dp.message(Command("stats"))
async def stats_cmd(m: Message):
    if m.from_user.id not in config.ADMINS:
        return
    s = await db.stats()
    await m.answer(
        f"<b>Пользователи:</b> {s['users']} (+{s['users_new_24h']} за сутки)\n"
        f"Активных за 7 дней: {s['active_7d']} · платных: {s['paid']} · "
        f"заблокировали бота: {s['blocked']}\n\n"
        f"<b>Расшифровок:</b> {s['jobs']} (за сутки {s['jobs_24h']}), "
        f"минут всего {int(s['minutes'])}\n\n"
        f"<b>Оплат:</b> {s['pay_count']} на {s['pay_sum'] / 100:.0f} ₽ "
        f"(за 30 дней {s['pay_sum_30d'] / 100:.0f} ₽)\n"
        f"<b>Звёздами:</b> {s['stars_count']} на {s['stars_sum']} ⭐\n"
        f"<b>Без чека:</b> {s['no_receipt']}",
        parse_mode="HTML",
    )


@dp.message(Command("export"))
async def export_cmd(m: Message):
    if m.from_user.id not in config.ADMINS:
        return
    stamp = time.strftime("%Y%m%d-%H%M")
    for name, data in (
        (f"users-{stamp}.csv", await db.export_users_csv()),
        (f"payments-{stamp}.csv", await db.export_payments_csv()),
        (f"events-{stamp}.csv", await db.export_events_csv()),
    ):
        await m.answer_document(BufferedInputFile(data, filename=name))
    await m.answer("Выгрузка готова. CSV с разделителем «;» — Excel откроет как есть.")


@dp.message(Command("evaltask"))
async def evaltask_cmd(m: Message, command: CommandObject, bot: Bot):
    """Прогон режима по сохранённым расшифровкам — без повторной транскрибации,
    только LLM. Нужен, чтобы правки промптов проверялись на реальном материале,
    а не на ощущениях: /evaltask protocol 3"""
    if m.from_user.id not in config.ADMINS:
        return

    parts = (command.args or "").split()
    task = parts[0] if parts else "protocol"
    try:
        count = int(parts[1]) if len(parts) > 1 else 3
    except ValueError:
        count = 3
    count = max(1, min(count, 10))

    jobs = await db.recent_jobs(count)
    if not jobs:
        return await m.answer("В базе нет расшифровок для прогона.")

    lines = [f"# Прогон режима «{task}» по {len(jobs)} записям",
             time.strftime("%d.%m.%Y %H:%M"), ""]
    total = time.time()

    async with working(m, f"Прогоняю «{task}» по {len(jobs)} записям"):
        for j in jobs:
            text = j["rawText"] or ""
            head = (f"## Запись #{j['id']} — {j['file_name']}\n"
                    f"{int(j['duration'] // 60)} мин · {len(text)} символов\n")
            if not text.strip():
                lines += [head, "_пустая расшифровка, пропущено_", ""]
                continue

            started = time.time()
            try:
                res = await rp.analyze(text, task=task, glossary="")
                took = time.time() - started
                lines += [head, f"_обработано за {took:.0f} с_", "", res, "", "---", ""]
            except Exception as e:
                lines += [head, f"**ОШИБКА:** {type(e).__name__}: {e}", "", "---", ""]

    lines.append(f"_Всего: {time.time() - total:.0f} с_")
    report = "\n".join(lines)

    await m.answer_document(
        BufferedInputFile(report.encode("utf-8"),
                          filename=f"eval-{task}-{time.strftime('%Y%m%d-%H%M')}.md"),
        caption=f"«{task}» по {len(jobs)} записям, {time.time() - total:.0f} с",
    )


@dp.message(Command("maintenance"))
async def maintenance_cmd(m: Message, command: CommandObject):
    if m.from_user.id not in config.ADMINS:
        return
    args = (command.args or "").strip()

    if args.lower() in ("off", "выкл"):
        await db.set_setting(MAINT_KEY, None)
        return await m.answer("Техработы выключены — бот принимает запросы.")

    if not args or args.lower() in ("on", "вкл"):
        text = ("Ведутся технические работы. Скоро всё заработает — "
                "пришлите файл чуть позже, ничего не потеряется.")
    else:
        text = args[3:].strip() if args.lower().startswith("on ") else args

    await db.set_setting(MAINT_KEY, text)
    await m.answer(f"Техработы включены. Пользователи увидят:\n\n🛠 {text}\n\n"
                   "Выключить: /maintenance off")


@dp.message(Command("broadcast"))
async def broadcast_cmd(m: Message, command: CommandObject, bot: Bot):
    if m.from_user.id not in config.ADMINS:
        return
    text = (command.args or "").strip()
    if not text:
        return await m.answer("Формат: /broadcast текст сообщения")

    ids = await db.all_user_ids()
    await m.answer(f"Рассылаю {len(ids)} пользователям…")
    ok = failed = 0
    for uid in ids:
        try:
            await bot.send_message(uid, text)
            ok += 1
        except Exception as e:
            failed += 1
            if "blocked" in str(e).lower() or "chat not found" in str(e).lower():
                await db.mark_blocked(uid)
        await asyncio.sleep(0.05)
    await m.answer(f"Готово: доставлено {ok}, не дошло {failed}.")


# ---------- приём медиа ----------
def _extract_media(m: Message):
    if m.voice:       return m.voice.file_id, "audio", "voice.ogg", m.voice.duration
    if m.audio:       return m.audio.file_id, "audio", (m.audio.file_name or "audio.mp3"), m.audio.duration
    if m.video:       return m.video.file_id, "video", (m.video.file_name or "video.mp4"), m.video.duration
    if m.video_note:  return m.video_note.file_id, "video", "circle.mp4", m.video_note.duration
    if m.document and (m.document.mime_type or "").startswith(("audio", "video")):
        ft = "video" if m.document.mime_type.startswith("video") else "audio"
        return m.document.file_id, ft, (m.document.file_name or "file"), 0
    return None


async def allowed_minutes(uid: int) -> tuple[int, str]:
    """Сколько минут можно потратить на одну запись прямо сейчас.
    Возвращает (минуты, причина отказа). Минуты 0 — обрабатывать нельзя."""
    plan = await db.effective_plan(uid)
    max_min = config.PLANS[plan]["max_file_min"]
    left = int(await db.remaining_minutes(uid))

    if left <= 0:
        return 0, ("Минуты в тарифе закончились. Лимит обновится в начале следующего "
                   "месяца — или можно перейти на тариф с большим запасом.")
    return min(max_min, left), ""


def _limit_message(err: str, allowed: int) -> str | None:
    """Воркер отказал по длительности — объясняем человеку, а не показываем трейсбек."""
    if "длиннее допустимого" in err or "too long" in err.lower():
        return (f"Запись длиннее, чем сейчас доступно — {allowed} мин.\n\n"
                "Это минимум из лимита тарифа на один файл и остатка минут за месяц. "
                "На старшем тарифе обработается целиком.")
    return None


_active_transcriptions: set[int] = set()


def one_transcription(fn):
    """One in-flight file/link per user in the single polling process."""
    @wraps(fn)
    async def wrapped(m, *args, **kwargs):
        uid = m.from_user.id
        # No await between check and add: atomic within this event loop.
        if uid in _active_transcriptions:
            return await m.answer("Предыдущая запись ещё обрабатывается. Дождитесь результата.")
        _active_transcriptions.add(uid)
        try:
            return await fn(m, *args, **kwargs)
        finally:
            _active_transcriptions.discard(uid)
    return wrapped


@dp.message(F.voice | F.audio | F.video | F.video_note | F.document)
@one_transcription
async def on_media(m: Message, bot: Bot, state: FSMContext):
    await state.clear()
    media = _extract_media(m)
    if not media:
        return await m.answer("Пришлите аудио или видео (или ссылку).")
    file_id, file_type, file_name, duration = media

    allowed, deny = await allowed_minutes(m.from_user.id)
    if not allowed:
        await db.log_event(m.from_user.id, "limit_min", "0 left")
        return await m.answer(deny, reply_markup=kb.buy_kb() if (config.PAYMENTS_ENABLED or config.STARS_ENABLED) else None)

    # Быстрый отказ, когда длительность известна (голосовые, аудио, видео).
    # Для документов Telegram её не сообщает — там потолок проверит воркер по ffprobe,
    # ещё до запуска распознавания, поэтому обойти лимит файлом-документом нельзя.
    if duration and duration / 60 > allowed:
        await db.log_event(m.from_user.id, "limit_len", f"{duration}s > {allowed}m")
        return await m.answer(
            f"На эту запись нужно ~{int(duration / 60)} мин, а сейчас доступно {allowed}.\n\n"
            "Это минимум из лимита тарифа на один файл и остатка минут за месяц.",
            reply_markup=kb.buy_kb() if (config.PAYMENTS_ENABLED or config.STARS_ENABLED) else None,
        )

    started = time.time()
    ok = False
    tmp_path: Path | None = None
    try:
        async with working(m, "Обрабатываю запись. Первый запуск после паузы дольше"):
            tmp_path = await fetch_to_disk(bot, file_id)
            # to_thread обязателен: boto3 блокирующий, без него на время заливки
            # встаёт весь бот — и polling, и счётчик в статус-сообщении.
            url = await asyncio.to_thread(storage.upload_file, tmp_path, file_name)
            _drop_tmp(tmp_path)   # диск освобождаем сразу, не ждём GPU
            tmp_path = None

            u = await db.get_user(m.from_user.id)
            options = {"language": "auto", "enableDiarization": True,
                       "customGlossary": u["glossary"] or ""}
            t = await rp.transcribe(url, file_name, file_type, options,
                                    max_duration_sec=allowed * 60)
            t["fileName"] = file_name

            mins = (t.get("durationEstimateSec") or duration or 0) / 60
            job_id = await db.save_job(m.from_user.id, file_name, t, minutes=mins)
            await db.log_event(m.from_user.id, "file",
                               f"{file_name} · {mins:.1f} мин · {time.time()-started:.0f}с")

        await m.answer(result_caption(t), parse_mode="HTML", reply_markup=kb.result_kb(job_id))
        ok = True
    except FileGoneError:
        # Локальный Bot API помнит, что файл уже скачан, и повторно его не тянет,
        # а мы удаляем файл сразу после чтения. Повторная обработка того же file_id
        # (пересылка прежнего сообщения, переигранный апдейт после рестарта службы)
        # приходит сюда: путь есть, файла нет.
        logging.warning("file gone: uid=%s file=%s", m.from_user.id, file_name)
        await db.log_event(m.from_user.id, "file_gone", file_name)
        return await m.answer(
            "Эта запись больше недоступна на сервере — скорее всего, она уже обрабатывалась.\n\n"
            "Пришлите файл заново как новое сообщение: пересылка прежнего не поможет, "
            "у неё тот же идентификатор файла."
        )
    except Exception as e:
        logging.exception("transcribe failed")
        await db.log_event(m.from_user.id, "error", f"{file_name}: {_safe_err(e)}")

        limit_msg = _limit_message(str(e), allowed)
        if limit_msg:
            return await m.answer(
                limit_msg, reply_markup=kb.buy_kb() if (config.PAYMENTS_ENABLED or config.STARS_ENABLED) else None)

        await m.answer(
            "Не получилось обработать запись. Мы уже знаем о проблеме и разбираемся.\n"
            f"Если срочно — напишите {SUPPORT}."
        )
        await notify_admins(bot, f"❗️ <b>Ошибка обработки файла</b>\n"
            f"Юзер: <code>{m.from_user.id}</code> (@{m.from_user.username or '—'})\n"
            f"Файл: {html.escape(file_name)} · {file_type} · {duration} сек\n"
            f"<pre>{html.escape(_safe_err(e))[:1500]}</pre>")
        return
    finally:
        # Сюда попадаем и на return из except: файл на диске не должен пережить хендлер.
        if tmp_path is not None:
            _drop_tmp(tmp_path)

    # Подсказка про глоссарий — вне try, чтобы её сбой не выглядел как провал расшифровки
    if ok:
        try:
            u2 = await db.get_user(m.from_user.id)
            if not (u2["glossary"] or "").strip() and (u2["files_count"] or 0) <= 3:
                from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
                await m.answer(
                    "💡 <b>Совет.</b> Если в записях звучат имена, бренды или "
                    "профессиональные термины — добавьте их в глоссарий. Распознавание "
                    "перестанет коверкать названия, а конспект и протокол станут точнее.",
                    parse_mode="HTML",
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                        InlineKeyboardButton(text="📚 Добавить термины",
                                             callback_data="set:glossary")]]),
                )
        except Exception:
            logging.exception("glossary hint failed")


# ---------- callbacks: инструменты ----------
async def callback_job(c: CallbackQuery, actions) -> tuple[dict, str] | None:
    try:
        _, jid, action = c.data.split(":")
        if action not in actions:
            raise ValueError("Unknown action")
        job = await db.get_job(int(jid), c.from_user.id)
    except (ValueError, AttributeError):
        job = None
    if not job:
        await c.answer("Задание не найдено", show_alert=True)
        return None
    return job, action


@dp.callback_query(F.data.startswith("t:"))
async def tool_cb(c: CallbackQuery, state: FSMContext):
    found = await callback_job(c, {*TASK_LABELS, "qa"})
    if not found:
        return
    job, task = found
    jid = job["id"]
    if task == "qa":
        await state.set_state(Flow.qa)
        await state.update_data(job_id=int(jid))
        await c.message.answer("Задайте вопрос по этой записи (или /cancel):")
        return await c.answer()
    await c.answer()
    await db.log_event(c.from_user.id, "tool", task)
    u = await db.get_user(c.from_user.id)
    text = job["transcript"].get("rawText", "")
    async with working(c.message, TASK_LABELS.get(task, "Обрабатываю")):
        res = await rp.analyze(text, task=task, glossary=u["glossary"] or "")
    await send_long(c.message, res, reply_markup=kb.result_kb(int(jid)))

@dp.message(Flow.qa, F.text)
async def qa_answer(m: Message, state: FSMContext):
    if is_interrupt(m):
        await state.clear()
        return await m.answer("Вопрос отменён. Повторите команду.", reply_markup=kb.main_menu)

    data = await state.get_data()
    await state.clear()
    job = await db.get_job(data.get("job_id"), m.from_user.id)
    if not job:
        return await m.answer("Задание не найдено.")

    await db.log_event(m.from_user.id, "qa", (m.text or "")[:120])
    u = await db.get_user(m.from_user.id)
    async with working(m, "Ищу ответ в записи"):
        res = await rp.analyze(job["transcript"].get("rawText", ""), task="qa",
                               question=m.text, glossary=u["glossary"] or "")
    await send_long(m, res, reply_markup=kb.result_kb(data["job_id"]))


# ---------- callbacks: перевод ----------
@dp.callback_query(F.data.startswith("tr:"))
async def translate_cb(c: CallbackQuery):
    found = await callback_job(c, {*config.LANGS, "menu"})
    if not found:
        return
    job, code = found
    jid = job["id"]
    if code == "menu":
        await c.answer()
        return await c.message.answer("Выберите язык перевода:",
                                      reply_markup=kb.translate_kb(int(jid)))

    await c.answer()
    lang = config.LANGS.get(code, code)
    await db.log_event(c.from_user.id, "translate", code)
    u = await db.get_user(c.from_user.id)
    async with working(c.message, f"Перевожу на {lang}"):
        res = await rp.analyze(job["transcript"].get("rawText", ""), task="translate",
                               target_lang=lang, glossary=u["glossary"] or "")
    await send_long(c.message, res, reply_markup=kb.result_kb(int(jid)))



# ---------- callbacks: экспорт ----------
@dp.callback_query(F.data.startswith("ex:"))
async def export_cb(c: CallbackQuery):
    found = await callback_job(c, {"menu", "srt", "txt", "md"})
    if not found:
        return
    job, fmt = found
    jid = job["id"]
    if fmt == "menu":
        await c.answer()
        return await c.message.answer("Формат файла:", reply_markup=kb.export_kb(int(jid)))
    t = job["transcript"]
    name = job["file_name"].rsplit(".", 1)[0]
    if fmt == "srt":
        content, fn = exporters.to_srt(t), f"{name}.srt"
    elif fmt == "txt":
        content, fn = exporters.to_txt(t), f"{name}.txt"
    else:
        content, fn = exporters.to_md(t, job["file_name"]), f"{name}.md"
    await c.answer()
    await db.log_event(c.from_user.id, "export", fmt)
    await c.message.answer_document(BufferedInputFile(content.encode("utf-8"), filename=fn),
                                    reply_markup=kb.result_kb(int(jid)))


# ---------- home / settings callbacks ----------
@dp.callback_query(F.data == "home")
async def home_cb(c: CallbackQuery, state: FSMContext):
    await state.clear()
    await c.answer()
    await c.message.answer("Главное меню 👇", reply_markup=kb.main_menu)


@dp.callback_query(F.data == "set:promo")
async def set_promo(c: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.promo)
    await c.message.answer("Введите промокод (или /cancel):")
    await c.answer()


@dp.callback_query(F.data == "set:glossary")
async def set_gloss(c: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.glossary)
    await c.message.answer(
        "Пришлите термины через запятую — имена людей, названия компаний и брендов, "
        "профессиональный сленг, аббревиатуры.\n\n"
        "Например: Hettich, Egger, Kuppersberg, Битрикс, СКЮ, оборачиваемость\n\n"
        "Отмена — /cancel"
    )
    await c.answer()


@dp.message(Flow.promo, F.text)
async def promo_input(m: Message, state: FSMContext):
    if is_interrupt(m):
        await state.clear()
        return await m.answer("Ввод промокода отменён. Повторите команду.",
                              reply_markup=kb.main_menu)
    await state.clear()
    ok, txt = await db.redeem_promo(m.from_user.id, m.text.strip().upper())
    await db.log_event(m.from_user.id, "promo", f"{m.text.strip().upper()} → {ok}")
    await m.answer(txt)


@dp.message(Flow.glossary, F.text)
async def gloss_input(m: Message, state: FSMContext):
    if is_interrupt(m):
        await state.clear()
        return await m.answer("Ввод глоссария отменён. Повторите команду.",
                              reply_markup=kb.main_menu)
    await state.clear()
    await db.set_glossary(m.from_user.id, m.text.strip())
    await m.answer("Глоссарий сохранён — учту его в следующих расшифровках.")


# ---------- расшифровка по ссылке ----------
@dp.message(F.text.regexp(r"https?://"))
@one_transcription
async def on_link(m: Message, state: FSMContext, bot: Bot):
    await state.clear()
    url = LINK_RE.search(m.text).group(0)

    allowed, deny = await allowed_minutes(m.from_user.id)
    if not allowed:
        await db.log_event(m.from_user.id, "limit_min", "link, 0 left")
        return await m.answer(deny, reply_markup=kb.buy_kb() if (config.PAYMENTS_ENABLED or config.STARS_ENABLED) else None)

    started = time.time()
    try:
        async with working(m, "Качаю запись по ссылке"):
            u = await db.get_user(m.from_user.id)
            options = {"language": "auto", "enableDiarization": True,
                       "customGlossary": u["glossary"] or ""}
            t = await rp.transcribe_url(url, options, max_duration_sec=allowed * 60)
            t["fileName"] = t.get("fileName") or "Запись по ссылке"

            mins = (t.get("durationEstimateSec") or 0) / 60
            job_id = await db.save_job(m.from_user.id, t["fileName"], t, minutes=mins)
            await db.log_event(m.from_user.id, "link",
                               f"{url[:80]} · {mins:.1f} мин · {time.time()-started:.0f}с")

        await m.answer(result_caption(t), parse_mode="HTML", reply_markup=kb.result_kb(job_id))
    except Exception as e:
        logging.exception("link transcribe failed")
        err = _safe_err(e)
        await db.log_event(m.from_user.id, "error", f"link {url[:80]}: {err}")

        limit_msg = _limit_message(err, allowed)
        if limit_msg:
            return await m.answer(
                limit_msg, reply_markup=kb.buy_kb() if (config.PAYMENTS_ENABLED or config.STARS_ENABLED) else None)

        blocked = any(s in err.lower() for s in
                      ("sign in", "confirm you", "403", "private", "unavailable",
                       "not a bot", "age", "geo"))
        await m.answer(
            "Площадка не отдала запись — видео закрытое, удалённое или требует входа "
            "в аккаунт.\n\nПопробуйте скачать файл и прислать его сюда."
            if blocked else
            "Не получилось обработать ссылку. Мы уже знаем о проблеме.\n"
            f"Если срочно — напишите {SUPPORT}."
        )
        await notify_admins(bot, f"❗️ <b>Ошибка по ссылке</b>\n"
            f"Юзер: <code>{m.from_user.id}</code> (@{m.from_user.username or '—'})\n"
            f"URL: {html.escape(url[:200])}\n"
            f"<pre>{html.escape(err)[:1200]}</pre>")


# ---------- заглушка: строго последний message-хендлер ----------
@dp.message()
async def fallback(m: Message):
    await m.answer(
        "Не понял команду. Пришлите аудио или видео файлом — расшифрую. "
        "Или откройте меню: /help",
        reply_markup=kb.main_menu,
    )


# ---------- глобальный перехват ошибок ----------
_last_err: dict[str, float] = {}


@dp.errors()
async def on_error(event: ErrorEvent, bot: Bot):
    exc = event.exception
    logging.exception("Update failed: %s", exc, exc_info=exc)

    sig = f"{type(exc).__name__}:{exc}"[:200]
    now = time.time()
    if now - _last_err.get(sig, 0) < 60:
        return True
    _last_err[sig] = now

    upd = event.update
    who = obj = "—"
    if upd.message:
        who = f"{upd.message.from_user.id} (@{upd.message.from_user.username or '—'})"
        obj = (upd.message.text or upd.message.content_type or "")[:200]
    elif upd.callback_query:
        who = f"{upd.callback_query.from_user.id} (@{upd.callback_query.from_user.username or '—'})"
        obj = upd.callback_query.data or ""

    tb = _safe_err(
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))[-1500:]
    await notify_admins(
        bot,
        f"🔥 <b>{html.escape(type(exc).__name__)}</b>\n"
        f"Юзер: <code>{who}</code>\n"
        f"Ввод: <code>{html.escape(str(obj))}</code>\n"
        f"<pre>{html.escape(tb)}</pre>",
    )
    return True


# ---------- запуск ----------
USER_COMMANDS = [
    BotCommand(command="start", description="Главное меню"),
    BotCommand(command="balance", description="Мой тариф и остаток минут"),
    BotCommand(command="tariffs", description="Тарифы и оплата"),
    BotCommand(command="settings", description="Промокод и глоссарий"),
    BotCommand(command="terms", description="Оферта и политика"),
    BotCommand(command="support", description="Поддержка"),
    BotCommand(command="help", description="Что я умею"),
]

ADMIN_COMMANDS = USER_COMMANDS + [
    BotCommand(command="stats", description="Статистика"),
    BotCommand(command="export", description="Выгрузка CSV"),
    BotCommand(command="pending", description="Оплаты без чека"),
    BotCommand(command="receipt", description="Отправить чек: /receipt ID URL"),
    BotCommand(command="newpromo", description="Создать промокод"),
    BotCommand(command="maintenance", description="Техработы on/off"),
    BotCommand(command="broadcast", description="Рассылка всем"),
    BotCommand(command="refundstars", description="Вернуть звёзды"),
    BotCommand(command="evaltask", description="Прогон промпта: /evaltask protocol 3"),
]


async def main():
    if config.TEST_SETTINGS is not None:
        raise SystemExit("Use integration_test.py for the restricted test mode")
    if not config.BOT_TOKEN:
        raise SystemExit("BOT_TOKEN не задан в .env")
    await db.init()
    _clean_tmp()
    session = None
    if getattr(config, "LOCAL_BOT_API", ""):
        # timeout по умолчанию 60 с — его не хватает: getFile в локальном режиме
        # ждёт, пока сервер выкачает файл из Telegram, а это для гигабайтного
        # видео минуты. Иначе аплоад падает по таймауту на ровном месте.
        session = AiohttpSession(
            api=TelegramAPIServer.from_base(config.LOCAL_BOT_API, is_local=True),
            timeout=int(getattr(config, "TG_TIMEOUT", 1800)))
        logging.info("Использую локальный Bot API: %s", config.LOCAL_BOT_API)

    bot = Bot(config.BOT_TOKEN, session=session,
              default=DefaultBotProperties(parse_mode=None))

    await bot.set_my_commands(USER_COMMANDS, scope=BotCommandScopeDefault())
    for admin in config.ADMINS:
        try:
            await bot.set_my_commands(ADMIN_COMMANDS, scope=BotCommandScopeChat(chat_id=admin))
        except Exception:
            logging.exception("set admin commands failed for %s", admin)

    if await db.get_setting(MAINT_KEY):
        logging.warning("Бот стартовал в режиме техработ — выключить: /maintenance off")

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
