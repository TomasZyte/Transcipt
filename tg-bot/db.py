"""SQLite (aiosqlite): пользователи, лимиты, задания, промокоды, платежи, события."""
import asyncio
import csv
import io
import json
import math
import time
from functools import wraps
import aiosqlite

import config

_db: aiosqlite.Connection | None = None
_lock = asyncio.Lock()
_owner: asyncio.Task | None = None


def serialized(fn):
    """Keep each operation on the shared connection in one task, including commits.

    Nested calls by the same task are allowed; another handler must not commit
    a payment/job transaction halfway through. Roll back failed writes as well.
    """
    @wraps(fn)
    async def wrapped(*args, **kwargs):
        global _owner
        task = asyncio.current_task()
        if _owner is task:
            return await fn(*args, **kwargs)
        async with _lock:
            _owner = task
            try:
                return await fn(*args, **kwargs)
            except BaseException:
                await _db.rollback()
                raise
            finally:
                _owner = None
    return wrapped


async def init(path: str = None):
    global _db, _lock, _owner
    _lock, _owner = asyncio.Lock(), None
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
            lang TEXT DEFAULT 'ru',            -- язык распознавания
            glossary TEXT DEFAULT '',
            created INTEGER
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            file_name TEXT,
            transcript TEXT,
            created INTEGER
        );
        CREATE TABLE IF NOT EXISTS promo (
            code TEXT PRIMARY KEY,
            plan TEXT,
            months INTEGER,
            uses_left INTEGER,
            created INTEGER
        );
        CREATE TABLE IF NOT EXISTS redemptions (
            user_id INTEGER, code TEXT,
            PRIMARY KEY (user_id, code)
        );
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            plan TEXT NOT NULL,
            amount INTEGER NOT NULL,           -- копейки
            currency TEXT DEFAULT 'RUB',
            charge_id TEXT NOT NULL UNIQUE,    -- telegram_payment_charge_id
            provider_charge_id TEXT,           -- id платежа в ЮKassa
            email TEXT,
            phone TEXT,
            name TEXT,
            receipt_url TEXT,
            receipt_sent INTEGER DEFAULT 0,
            is_test INTEGER DEFAULT 0,
            created INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_payments_user ON payments(user_id);
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            kind TEXT,                         -- start / file / tool / pay / error ...
            detail TEXT,
            created INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id);
        CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, created);
        """
    )
    await _migrate()
    await _db.commit()


async def _migrate():
    """Дозаливка колонок в существующую базу — ALTER TABLE не умеет IF NOT EXISTS."""
    async def add_columns(table: str, cols: dict[str, str]):
        cur = await _db.execute(f"PRAGMA table_info({table})")
        have = {r["name"] for r in await cur.fetchall()}
        for name, ddl in cols.items():
            if name not in have:
                await _db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    await add_columns("users", {
        "username":   "TEXT",
        "first_name": "TEXT",
        "last_name":  "TEXT",
        "lang_code":  "TEXT",           # язык интерфейса Telegram
        "is_premium": "INTEGER DEFAULT 0",
        "source":     "TEXT",           # метка из deep-link /start?start=...
        "first_seen": "INTEGER",
        "last_seen":  "INTEGER",
        "msg_count":  "INTEGER DEFAULT 0",
        "files_count": "INTEGER DEFAULT 0",
        "email":      "TEXT",           # из последней оплаты
        "blocked":    "INTEGER DEFAULT 0",
    })
    await add_columns("payments", {
        "phone": "TEXT",
        "name":  "TEXT",
    })


def _month() -> str:
    t = time.gmtime()
    return f"{t.tm_year}-{t.tm_mon:02d}"


@serialized
async def get_user(uid: int) -> aiosqlite.Row:
    cur = await _db.execute("SELECT * FROM users WHERE id=?", (uid,))
    row = await cur.fetchone()
    if row is None:
        now = int(time.time())
        await _db.execute(
            "INSERT INTO users(id, period, created, first_seen, last_seen) VALUES(?,?,?,?,?)",
            (uid, _month(), now, now, now))
        await _db.commit()
        cur = await _db.execute("SELECT * FROM users WHERE id=?", (uid,))
        row = await cur.fetchone()
    return row


@serialized
async def user_exists(uid: int) -> bool:
    cur = await _db.execute("SELECT 1 FROM users WHERE id=?", (uid,))
    return await cur.fetchone() is not None


@serialized
async def touch_user(uid: int, username=None, first_name=None, last_name=None,
                     lang_code=None, is_premium=False, source=None, is_file=False):
    """Обновляет профиль и счётчики на каждом действии. source пишется только один раз."""
    await get_user(uid)
    now = int(time.time())
    await _db.execute(
        """UPDATE users SET
             username=?, first_name=?, last_name=?, lang_code=?, is_premium=?,
             last_seen=?, msg_count=COALESCE(msg_count,0)+1,
             files_count=COALESCE(files_count,0)+?,
             first_seen=COALESCE(first_seen, ?),
             source=COALESCE(source, ?)
           WHERE id=?""",
        (username, first_name, last_name, lang_code, int(bool(is_premium)),
         now, int(bool(is_file)), now, source, uid),
    )
    await _db.commit()


@serialized
async def log_event(uid: int, kind: str, detail: str = ""):
    await _db.execute(
        "INSERT INTO events(user_id, kind, detail, created) VALUES(?,?,?,?)",
        (uid, kind, detail[:500], int(time.time())),
    )
    await _db.commit()


@serialized
async def effective_plan(uid: int) -> str:
    u = await get_user(uid)
    if u["plan"] != "free" and u["plan_until"] and u["plan_until"] < time.time():
        await _db.execute("UPDATE users SET plan='free' WHERE id=?", (uid,))
        await _db.commit()
        return "free"
    return u["plan"]


@serialized
async def remaining_minutes(uid: int) -> float:
    u = await get_user(uid)
    plan = await effective_plan(uid)
    used = u["minutes_used"] if u["period"] == _month() else 0.0
    limit = config.PLANS[plan]["minutes"]
    return max(0.0, limit - used)


@serialized
async def add_usage(uid: int, minutes: float):
    await get_user(uid)
    await _add_usage(uid, minutes)
    await _db.commit()


async def _add_usage(uid: int, minutes: float):
    if not math.isfinite(minutes) or minutes < 0:
        raise ValueError("Usage must be a finite nonnegative number")
    month = _month()
    await _db.execute(
        "UPDATE users SET minutes_used=(CASE WHEN period=? THEN minutes_used ELSE 0 END)+?, "
        "period=? WHERE id=?", (month, minutes, month, uid))


@serialized
async def set_plan(uid: int, plan: str, months: int = 1):
    await get_user(uid)
    until = int(time.time()) + months * 30 * 86400
    await _db.execute("UPDATE users SET plan=?, plan_until=? WHERE id=?", (plan, until, uid))
    await _db.commit()


@serialized
async def set_glossary(uid: int, text: str):
    await _db.execute("UPDATE users SET glossary=? WHERE id=?", (text, uid))
    await _db.commit()


@serialized
async def save_job(uid: int, file_name: str, transcript: dict,
                   minutes: float = 0) -> int:
    """Save the result and its usage in the same transaction."""
    await get_user(uid)
    cur = await _db.execute(
        "INSERT INTO jobs(user_id, file_name, transcript, created) VALUES(?,?,?,?)",
        (uid, file_name, json.dumps(transcript, ensure_ascii=False), int(time.time())),
    )
    await _add_usage(uid, minutes)
    await _db.commit()
    return cur.lastrowid


@serialized
async def get_job(job_id: int, uid: int) -> dict | None:
    cur = await _db.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, uid))
    row = await cur.fetchone()
    if not row:
        return None
    return {"id": row["id"], "user_id": row["user_id"], "file_name": row["file_name"],
            "transcript": json.loads(row["transcript"])}


# --- промокоды ---
@serialized
async def create_promo(code: str, plan: str, months: int, uses: int):
    await _db.execute(
        "INSERT OR REPLACE INTO promo(code, plan, months, uses_left, created) VALUES(?,?,?,?,?)",
        (code, plan, months, uses, int(time.time())),
    )
    await _db.commit()


@serialized
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


# --- платежи ---
@serialized
async def payment_exists(charge_id: str) -> bool:
    cur = await _db.execute("SELECT 1 FROM payments WHERE charge_id=?", (charge_id,))
    return await cur.fetchone() is not None


@serialized
async def apply_payment(uid: int, plan: str, amount: int, currency: str,
                      charge_id: str, provider_charge_id: str | None,
                      email: str | None, is_test: bool,
                      phone: str | None = None, name: str | None = None) -> tuple[int, int] | None:
    """Record charge and grant a month atomically; duplicates return None.

    Monthly usage is independent of subscription renewal and is not reset here.
    The serialized wrapper rolls back both changes on error/cancellation.
    """
    if plan not in config.PRICES:
        raise ValueError("Unknown paid plan")
    await get_user(uid)
    await _db.execute("BEGIN IMMEDIATE")
    cur = await _db.execute(
        """INSERT INTO payments(user_id, plan, amount, currency, charge_id,
                                provider_charge_id, email, phone, name, is_test, created)
           VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(charge_id) DO NOTHING""",
        (uid, plan, amount, currency, charge_id, provider_charge_id,
         email, phone, name, int(is_test), int(time.time())),
    )
    if not cur.rowcount:
        await _db.rollback()
        return None
    pid = cur.lastrowid
    u = await get_user(uid)
    now = int(time.time())

    if u["plan"] == plan and u["plan_until"] == 0 and plan != "free":
        until = 0            # бессрочный тариф с промокода не понижаем
    else:
        base = u["plan_until"] if (u["plan"] == plan and u["plan_until"] > now) else now
        until = base + config.SUB_DAYS * 86400

    await _db.execute(
        "UPDATE users SET plan=?, plan_until=?, email=COALESCE(?, email) WHERE id=?",
        (plan, until, email, uid),
    )
    await _db.commit()
    return pid, until


@serialized
async def set_receipt(payment_id: int, url: str) -> aiosqlite.Row | None:
    cur = await _db.execute("SELECT * FROM payments WHERE id=?", (payment_id,))
    row = await cur.fetchone()
    if not row:
        return None
    await _db.execute(
        "UPDATE payments SET receipt_url=?, receipt_sent=0 WHERE id=?", (url, payment_id)
    )
    await _db.commit()
    return row


@serialized
async def mark_receipt_sent(payment_id: int, url: str):
    await _db.execute("UPDATE payments SET receipt_sent=1 WHERE id=? AND receipt_url=?",
                      (payment_id, url))
    await _db.commit()


@serialized
async def pending_receipts() -> list[aiosqlite.Row]:
    cur = await _db.execute(
        "SELECT * FROM payments WHERE receipt_sent=0 AND is_test=0 ORDER BY id"
    )
    return await cur.fetchall()


@serialized
async def recent_jobs(limit: int = 5) -> list:
    """Последние расшифровки для прогона промптов без повторной транскрибации."""
    cur = await _db.execute(
        "SELECT id, user_id, file_name, transcript, created FROM jobs "
        "ORDER BY id DESC LIMIT ?", (limit,))
    rows = await cur.fetchall()
    out = []
    for r in rows:
        try:
            t = json.loads(r["transcript"])
        except Exception:
            continue
        out.append({"id": r["id"], "file_name": r["file_name"], "created": r["created"],
                    "rawText": t.get("rawText", ""),
                    "duration": t.get("durationEstimateSec", 0)})
    return out


# --- настройки ---
@serialized
async def get_setting(key: str, default: str | None = None) -> str | None:
    cur = await _db.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = await cur.fetchone()
    return row["value"] if row else default


@serialized
async def set_setting(key: str, value: str | None):
    if value is None:
        await _db.execute("DELETE FROM settings WHERE key=?", (key,))
    else:
        await _db.execute(
            "INSERT OR REPLACE INTO settings(key, value) VALUES(?,?)", (key, value)
        )
    await _db.commit()


# --- статистика, рассылка, выгрузка ---
@serialized
async def all_user_ids() -> list[int]:
    cur = await _db.execute("SELECT id FROM users WHERE COALESCE(blocked,0)=0 ORDER BY id")
    return [r["id"] for r in await cur.fetchall()]


@serialized
async def mark_blocked(uid: int):
    await _db.execute("UPDATE users SET blocked=1 WHERE id=?", (uid,))
    await _db.commit()


@serialized
async def count_users() -> int:
    cur = await _db.execute("SELECT COUNT(*) FROM users")
    row = await cur.fetchone()
    return row[0] if row else 0


@serialized
async def stats() -> dict:
    async def one(sql: str, *args):
        cur = await _db.execute(sql, args)
        row = await cur.fetchone()
        return row[0] if row else 0

    day = int(time.time()) - 86400
    week = int(time.time()) - 7 * 86400
    return {
        "users": await one("SELECT COUNT(*) FROM users"),
        "users_new_24h": await one("SELECT COUNT(*) FROM users WHERE created > ?", day),
        "active_7d": await one("SELECT COUNT(*) FROM users WHERE last_seen > ?", week),
        "blocked": await one("SELECT COUNT(*) FROM users WHERE COALESCE(blocked,0)=1"),
        "paid": await one("SELECT COUNT(*) FROM users WHERE plan != 'free'"),
        "jobs": await one("SELECT COUNT(*) FROM jobs"),
        "jobs_24h": await one("SELECT COUNT(*) FROM jobs WHERE created > ?", day),
        "minutes": await one("SELECT COALESCE(SUM(minutes_used), 0) FROM users"),
        # рубли и звёзды считаем раздельно: у XTR amount — это штуки, а не копейки
        "pay_count": await one(
            "SELECT COUNT(*) FROM payments WHERE is_test=0 AND currency='RUB'"),
        "pay_sum": await one(
            "SELECT COALESCE(SUM(amount),0) FROM payments WHERE is_test=0 AND currency='RUB'"),
        "pay_sum_30d": await one(
            "SELECT COALESCE(SUM(amount),0) FROM payments "
            "WHERE is_test=0 AND currency='RUB' AND created > ?",
            int(time.time()) - 30 * 86400),
        "stars_count": await one("SELECT COUNT(*) FROM payments WHERE currency='XTR'"),
        "stars_sum": await one(
            "SELECT COALESCE(SUM(amount),0) FROM payments WHERE currency='XTR'"),
        "no_receipt": await one(
            "SELECT COUNT(*) FROM payments WHERE is_test=0 AND receipt_sent=0"),
    }


def _rows_to_csv(rows, headers: list[str]) -> bytes:
    """CSV с BOM — Excel открывает кириллицу без плясок с кодировкой."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(headers)
    for r in rows:
        w.writerow(["" if r[h] is None else r[h] for h in headers])
    return buf.getvalue().encode("utf-8-sig")


@serialized
async def export_users_csv() -> bytes:
    cur = await _db.execute(
        """SELECT id, username, first_name, last_name, lang_code, is_premium, source,
                  plan, plan_until, minutes_used, period, email, msg_count, files_count,
                  blocked, first_seen, last_seen, created
           FROM users ORDER BY id"""
    )
    rows = await cur.fetchall()
    return _rows_to_csv(rows, [
        "id", "username", "first_name", "last_name", "lang_code", "is_premium", "source",
        "plan", "plan_until", "minutes_used", "period", "email", "msg_count", "files_count",
        "blocked", "first_seen", "last_seen", "created",
    ])


@serialized
async def export_payments_csv() -> bytes:
    cur = await _db.execute(
        """SELECT p.id, p.user_id, u.username, p.plan, p.amount, p.currency,
                  p.email, p.phone, p.name, p.charge_id, p.provider_charge_id,
                  p.receipt_url, p.receipt_sent, p.is_test, p.created
           FROM payments p LEFT JOIN users u ON u.id = p.user_id ORDER BY p.id"""
    )
    rows = await cur.fetchall()
    return _rows_to_csv(rows, [
        "id", "user_id", "username", "plan", "amount", "currency",
        "email", "phone", "name", "charge_id", "provider_charge_id",
        "receipt_url", "receipt_sent", "is_test", "created",
    ])


@serialized
async def export_events_csv(limit: int = 20000) -> bytes:
    cur = await _db.execute(
        """SELECT e.id, e.user_id, u.username, e.kind, e.detail, e.created
           FROM events e LEFT JOIN users u ON u.id = e.user_id
           ORDER BY e.id DESC LIMIT ?""", (limit,)
    )
    rows = await cur.fetchall()
    return _rows_to_csv(rows, ["id", "user_id", "username", "kind", "detail", "created"])
