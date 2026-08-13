"""SQLite (aiosqlite): пользователи, лимиты, задания-расшифровки, промокоды."""
import json
import time
import aiosqlite

import config

_db: aiosqlite.Connection | None = None


async def init(path: str = None):
    global _db
    _db = await aiosqlite.connect(path or config.DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            plan TEXT DEFAULT 'free',
            plan_until INTEGER DEFAULT 0,      -- unix, 0 = бессрочно/нет
            minutes_used REAL DEFAULT 0,
            period TEXT DEFAULT '',            -- 'YYYY-MM' текущего расхода
            lang TEXT DEFAULT 'ru',            -- язык распознавания (auto по умолчанию сверху)
            glossary TEXT DEFAULT '',
            created INTEGER
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            file_name TEXT,
            transcript TEXT,                   -- JSON {segments, rawText, srtContent, summary, ...}
            created INTEGER
        );
        CREATE TABLE IF NOT EXISTS promo (
            code TEXT PRIMARY KEY,
            plan TEXT,
            months INTEGER,                    -- 0 = бессрочно
            uses_left INTEGER,                 -- -1 = безлимит активаций
            created INTEGER
        );
        CREATE TABLE IF NOT EXISTS redemptions (
            user_id INTEGER, code TEXT,
            PRIMARY KEY (user_id, code)
        );
        """
    )
    await _db.commit()


def _month() -> str:
    # без Date.now в песочнице нельзя, но в проде time.time() доступен
    t = time.gmtime()
    return f"{t.tm_year}-{t.tm_mon:02d}"


async def get_user(uid: int) -> aiosqlite.Row:
    cur = await _db.execute("SELECT * FROM users WHERE id=?", (uid,))
    row = await cur.fetchone()
    if row is None:
        await _db.execute("INSERT INTO users(id, period, created) VALUES(?,?,?)",
                          (uid, _month(), int(time.time())))
        await _db.commit()
        cur = await _db.execute("SELECT * FROM users WHERE id=?", (uid,))
        row = await cur.fetchone()
    return row


async def effective_plan(uid: int) -> str:
    u = await get_user(uid)
    if u["plan"] != "free" and u["plan_until"] and u["plan_until"] < time.time():
        await _db.execute("UPDATE users SET plan='free' WHERE id=?", (uid,))
        await _db.commit()
        return "free"
    return u["plan"]


async def remaining_minutes(uid: int) -> float:
    u = await get_user(uid)
    plan = await effective_plan(uid)
    used = u["minutes_used"] if u["period"] == _month() else 0.0
    limit = config.PLANS[plan]["minutes"]
    return max(0.0, limit - used)


async def add_usage(uid: int, minutes: float):
    u = await get_user(uid)
    used = u["minutes_used"] if u["period"] == _month() else 0.0
    await _db.execute("UPDATE users SET minutes_used=?, period=? WHERE id=?",
                      (used + minutes, _month(), uid))
    await _db.commit()


async def set_plan(uid: int, plan: str, months: int = 1):
    await get_user(uid)
    until = int(time.time()) + months * 30 * 86400
    await _db.execute("UPDATE users SET plan=?, plan_until=? WHERE id=?", (plan, until, uid))
    await _db.commit()


async def set_glossary(uid: int, text: str):
    await _db.execute("UPDATE users SET glossary=? WHERE id=?", (text, uid))
    await _db.commit()


async def save_job(uid: int, file_name: str, transcript: dict) -> int:
    cur = await _db.execute(
        "INSERT INTO jobs(user_id, file_name, transcript, created) VALUES(?,?,?,?)",
        (uid, file_name, json.dumps(transcript, ensure_ascii=False), int(time.time())),
    )
    await _db.commit()
    return cur.lastrowid


async def get_job(job_id: int) -> dict | None:
    cur = await _db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
    row = await cur.fetchone()
    if not row:
        return None
    return {"id": row["id"], "user_id": row["user_id"], "file_name": row["file_name"],
            "transcript": json.loads(row["transcript"])}


# --- промокоды ---
async def create_promo(code: str, plan: str, months: int, uses: int):
    await _db.execute(
        "INSERT OR REPLACE INTO promo(code, plan, months, uses_left, created) VALUES(?,?,?,?,?)",
        (code, plan, months, uses, int(time.time())),
    )
    await _db.commit()


async def redeem_promo(uid: int, code: str) -> tuple[bool, str]:
    cur = await _db.execute("SELECT * FROM promo WHERE code=?", (code,))
    p = await cur.fetchone()
    if not p:
        return False, "Промокод не найден."
    if p["uses_left"] == 0:
        return False, "Промокод исчерпан."
    cur = await _db.execute("SELECT 1 FROM redemptions WHERE user_id=? AND code=?", (uid, code))
    if await cur.fetchone():
        return False, "Вы уже активировали этот промокод."
    until = 0 if p["months"] == 0 else int(time.time()) + p["months"] * 30 * 86400
    await _db.execute("UPDATE users SET plan=?, plan_until=? WHERE id=?", (p["plan"], until, uid))
    await _db.execute("INSERT INTO redemptions(user_id, code) VALUES(?,?)", (uid, code))
    if p["uses_left"] > 0:
        await _db.execute("UPDATE promo SET uses_left=uses_left-1 WHERE code=?", (code,))
    await _db.commit()
    title = config.PLANS[p["plan"]]["title"]
    term = "бессрочно" if p["months"] == 0 else f"на {p['months']} мес."
    return True, f"Активирован тариф «{title}» {term}. 🎉"