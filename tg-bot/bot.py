"""Точка входа Telegram-бота транскрибации (aiogram 3)."""
import asyncio
import io
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, CallbackQuery, BufferedInputFile

import config
import db
import keyboards as kb
import runpod_client as rp
import storage
import exporters

logging.basicConfig(level=logging.INFO)
dp = Dispatcher(storage=MemoryStorage())


class Flow(StatesGroup):
    qa = State()
    promo = State()
    glossary = State()


# ---------- утилиты ----------
async def send_long(msg: Message, text: str, **kw):
    for i in range(0, len(text), 4000):
        await msg.answer(text[i:i + 4000], **kw)


def result_caption(t: dict) -> str:
    s = t.get("summary", {}) or {}
    speakers = len({seg.get("speaker") for seg in t.get("segments", [])})
    dur = int(t.get("durationEstimateSec", 0))
    head = (f"✅ <b>{t.get('fileName','Запись')}</b>\n"
            f"Язык: {t.get('languageDetected','?')} · Спикеров: {speakers} · "
            f"{dur//60} мин {dur%60} сек\n")
    if s.get("overview"):
        head += f"\n<b>Кратко:</b> {s['overview']}\n"
    if s.get("keyPoints"):
        head += "\n<b>Тезисы:</b>\n" + "\n".join(f"• {x}" for x in s["keyPoints"][:6])
    return head[:4000]


# ---------- команды/меню ----------
@dp.message(CommandStart())
async def start(m: Message):
    await db.get_user(m.from_user.id)
    await m.answer(
        "Привет! Я превращаю аудио и видео в текст — с разделением по спикерам, "
        "конспектом, протоколом и задачами. И всё это на приватном сервере: "
        "ваши записи не уходят в чужие облака.\n\n"
        "Пришлите голосовое, аудио или видео — обработаю сразу.",
        reply_markup=kb.main_menu,
    )


@dp.message(F.text == "🎙 Новая транскрибация")
async def new_tr(m: Message):
    await m.answer("Пришлите аудио, голосовое или видео файлом или сообщением 👇")


@dp.message(F.text == "Тарифы")
async def tariffs(m: Message):
    plan = await db.effective_plan(m.from_user.id)
    left = await db.remaining_minutes(m.from_user.id)
    await m.answer(
        f"Ваш тариф: <b>{config.PLANS[plan]['title']}</b> · осталось "
        f"{int(left)} мин в этом месяце\n\n"
        "<b>Бесплатный</b> — 90 мин/мес, разделение по спикерам, конспект, экспорт\n"
        "<b>Старт — 590 ₽/мес</b> — 20 ч/мес, протокол, задачи, цифры, все форматы, файлы до 2 ч\n"
        "<b>Про — 1290 ₽/мес</b> — 60 ч/мес + перевод, Q&A, ссылки, глоссарий, файлы до 4 ч\n"
        "<b>Бизнес</b> — приватный инстанс и интеграции, по договорённости\n\n"
        "Есть промокод? Настройки → «Ввести промокод».",
        parse_mode="HTML",
    )


@dp.message(F.text == "Узнать больше")
async def about(m: Message):
    await m.answer(
        "Чем я отличаюсь от других сервисов:\n\n"
        "🔒 <b>Приватность</b> — записи не уходят в чужие облака, всё на нашем сервере.\n"
        "👥 <b>Разделение по спикерам</b> — на всех тарифах, без лимитов.\n"
        "🎯 <b>Точные тайм-коды</b> — идеальные субтитры.\n"
        "🧰 <b>Инструменты встреч</b> — протокол, задачи, цифры/бюджет, вопрос к AI.\n"
        "📚 <b>Глоссарий</b> — распознаю ваши термины, бренды и имена.",
        parse_mode="HTML",
    )


@dp.message(F.text == "Настройки")
async def settings(m: Message):
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    await m.answer("Настройки:", reply_markup=InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎁 Ввести промокод", callback_data="set:promo")],
        [InlineKeyboardButton(text="📚 Глоссарий терминов", callback_data="set:glossary")],
    ]))


@dp.message(Command("promo"))
async def promo_cmd(m: Message, state: FSMContext):
    parts = m.text.split(maxsplit=1)
    if len(parts) == 2:
        ok, txt = await db.redeem_promo(m.from_user.id, parts[1].strip().upper())
        await m.answer(txt)
    else:
        await state.set_state(Flow.promo)
        await m.answer("Введите промокод:")


# админ: создать промокод  /newpromo CODE plan months uses
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


@dp.message(F.voice | F.audio | F.video | F.video_note | F.document)
async def on_media(m: Message, bot: Bot):
    media = _extract_media(m)
    if not media:
        return await m.answer("Пришлите аудио или видео (или ссылку).")
    file_id, file_type, file_name, duration = media

    plan = await db.effective_plan(m.from_user.id)
    max_min = config.PLANS[plan]["max_file_min"]
    if duration and duration > max_min * 60:
        return await m.answer(f"Файл длиннее {max_min} мин — доступно на более высоком тарифе. "
                              "Смотрите «Тарифы».")
    left = await db.remaining_minutes(m.from_user.id)
    if duration and duration / 60 > left:
        return await m.answer(f"Не хватает минут в тарифе (осталось {int(left)}). Смотрите «Тарифы».")

    status = await m.answer("⏳ Обрабатываю запись… Первый запуск может занять пару минут.")
    try:
        tg_file = await bot.get_file(file_id)
        buf = io.BytesIO()
        await bot.download_file(tg_file.file_path, buf)
        url = storage.upload_bytes(buf.getvalue(), file_name)

        u = await db.get_user(m.from_user.id)
        options = {"language": "auto", "enableDiarization": True,
                   "customGlossary": u["glossary"] or ""}
        t = await rp.transcribe(url, file_name, file_type, options)
        t["fileName"] = file_name

        job_id = await db.save_job(m.from_user.id, file_name, t)
        await db.add_usage(m.from_user.id, (t.get("durationEstimateSec") or duration or 0) / 60)

        await status.delete()
        await m.answer(result_caption(t), parse_mode="HTML", reply_markup=kb.result_kb(job_id))
    except Exception as e:
        logging.exception("transcribe failed")
        await status.edit_text(f"Ошибка обработки: {e}")


# ---------- callbacks: инструменты ----------
@dp.callback_query(F.data.startswith("t:"))
async def tool_cb(c: CallbackQuery, state: FSMContext):
    _, jid, task = c.data.split(":")
    job = await db.get_job(int(jid))
    if not job:
        return await c.answer("Задание не найдено", show_alert=True)
    if task == "qa":
        await state.set_state(Flow.qa)
        await state.update_data(job_id=int(jid))
        await c.message.answer("Задайте вопрос по этой записи:")
        return await c.answer()
    await c.answer("Обрабатываю…")
    text = job["transcript"].get("rawText", "")
    res = await rp.analyze(text, task=task, glossary="")
    await send_long(c.message, res)


@dp.message(Flow.qa)
async def qa_answer(m: Message, state: FSMContext):
    data = await state.get_data()
    await state.clear()
    job = await db.get_job(data["job_id"])
    if not job:
        return await m.answer("Задание не найдено.")
    wait = await m.answer("💬 Думаю…")
    res = await rp.analyze(job["transcript"].get("rawText", ""), task="qa", question=m.text)
    await wait.delete()
    await send_long(m, res)


# ---------- callbacks: перевод ----------
@dp.callback_query(F.data.startswith("tr:"))
async def translate_cb(c: CallbackQuery):
    _, jid, code = c.data.split(":")
    if code == "menu":
        return await c.message.edit_reply_markup(reply_markup=kb.translate_kb(int(jid)))
    job = await db.get_job(int(jid))
    if not job:
        return await c.answer("Задание не найдено", show_alert=True)
    await c.answer(f"Перевод на {config.LANGS.get(code, code)}…")
    res = await rp.analyze(job["transcript"].get("rawText", ""), task="translate",
                           target_lang=config.LANGS.get(code, code))
    await send_long(c.message, res)


# ---------- callbacks: экспорт ----------
@dp.callback_query(F.data.startswith("ex:"))
async def export_cb(c: CallbackQuery):
    _, jid, fmt = c.data.split(":")
    if fmt == "menu":
        return await c.message.edit_reply_markup(reply_markup=kb.export_kb(int(jid)))
    job = await db.get_job(int(jid))
    if not job:
        return await c.answer("Задание не найдено", show_alert=True)
    t = job["transcript"]
    name = job["file_name"].rsplit(".", 1)[0]
    if fmt == "srt":
        content, fn = exporters.to_srt(t), f"{name}.srt"
    elif fmt == "txt":
        content, fn = exporters.to_txt(t), f"{name}.txt"
    else:
        content, fn = exporters.to_md(t, job["file_name"]), f"{name}.md"
    await c.answer()
    await c.message.answer_document(BufferedInputFile(content.encode("utf-8"), filename=fn))


# ---------- settings callbacks ----------
@dp.callback_query(F.data == "set:promo")
async def set_promo(c: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.promo)
    await c.message.answer("Введите промокод:")
    await c.answer()


@dp.callback_query(F.data == "set:glossary")
async def set_gloss(c: CallbackQuery, state: FSMContext):
    await state.set_state(Flow.glossary)
    await c.message.answer("Пришлите список терминов через запятую (имена, бренды, сленг):")
    await c.answer()


@dp.message(Flow.promo)
async def promo_input(m: Message, state: FSMContext):
    await state.clear()
    ok, txt = await db.redeem_promo(m.from_user.id, m.text.strip().upper())
    await m.answer(txt)


@dp.message(Flow.glossary)
async def gloss_input(m: Message, state: FSMContext):
    await state.clear()
    await db.set_glossary(m.from_user.id, m.text.strip())
    await m.answer("Глоссарий сохранён — учту его в следующих расшифровках.")


async def main():
    if not config.BOT_TOKEN:
        raise SystemExit("BOT_TOKEN не задан в .env")
    await db.init()
    bot = Bot(config.BOT_TOKEN, default=DefaultBotProperties(parse_mode=None))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
