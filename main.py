"""TikAi backend: FastAPI + SQLite. Run: uvicorn backend.main:app --port 8000"""
import hashlib, hmac, json, os, sqlite3, time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import parse_qsl

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
ALLOWED = {int(x) for x in os.environ.get("ALLOWED_USER_IDS", "").split(",") if x.strip()}
DB_PATH = os.environ.get("DB_PATH", "tikai.db")
INITDATA_MAX_AGE = 3600  # seconds

INTEGRATIONS = [  # id, label, env var holding the secret
    ("tiktok", "TikTok account", "TIKTOK_ACCESS_TOKEN"),
    ("ai_provider", "AI provider", "AI_API_KEY"),
    ("video_gen", "Video generation", "VIDEO_API_KEY"),
    ("tts", "Voice-over", "TTS_API_KEY"),
]
REQUIRED = {
    "planning": [],
    "production": ["ai_provider", "video_gen", "tts"],
    "autonomous": ["ai_provider", "video_gen", "tts", "tiktok", "publisher"],
}

app = FastAPI(title="TikAi API")


# ---------- storage ----------
@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    with db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS kv(user_id INT, k TEXT, v TEXT, PRIMARY KEY(user_id,k));
        CREATE TABLE IF NOT EXISTS videos(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INT, title TEXT,
            status TEXT, created_at TEXT, scheduled_at TEXT, published_at TEXT, cover_url TEXT,
            script TEXT, caption TEXT, metrics TEXT);
        CREATE TABLE IF NOT EXISTS video_events(id INTEGER PRIMARY KEY AUTOINCREMENT, video_id INT, ts TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS actions(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INT, ts TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INT, ts TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS plans(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INT, ts TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS experiments(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INT, ts TEXT, text TEXT, result TEXT);
        CREATE TABLE IF NOT EXISTS insights(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INT, ts TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS daily(user_id INT, date TEXT, views INT, followers INT, PRIMARY KEY(user_id,date));
        """)


init_db()
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def kv_get(uid, k, default=None):
    with db() as c:
        r = c.execute("SELECT v FROM kv WHERE user_id=? AND k=?", (uid, k)).fetchone()
    return json.loads(r["v"]) if r else default


def kv_set(uid, k, v):
    with db() as c:
        c.execute("INSERT INTO kv VALUES(?,?,?) ON CONFLICT(user_id,k) DO UPDATE SET v=excluded.v",
                  (uid, k, json.dumps(v)))


def log_action(uid, text):
    with db() as c:
        c.execute("INSERT INTO actions(user_id,ts,text) VALUES(?,?,?)", (uid, now(), text))


# ---------- Telegram initData verification ----------
def verify_init_data(init_data: str) -> dict:
    if not BOT_TOKEN:
        raise HTTPException(500, "BOT_TOKEN is not configured on the server")
    pairs = dict(parse_qsl(init_data, keep_blank_values=True))
    received = pairs.pop("hash", None)
    if not received:
        raise HTTPException(401, "Missing Telegram signature")
    check = "\n".join(f"{k}={v}" for k, v in sorted(pairs.items()))
    secret = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
    calc = hmac.new(secret, check.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(calc, received):
        raise HTTPException(401, "Invalid Telegram signature")
    if time.time() - int(pairs.get("auth_date", 0)) > INITDATA_MAX_AGE:
        raise HTTPException(401, "Session expired. Reopen the app from Telegram.")
    return json.loads(pairs["user"])


def current_user(authorization: str = Header(default="")) -> int:
    if not authorization.startswith("tma "):
        raise HTTPException(401, "Open this app from Telegram")
    uid = int(verify_init_data(authorization[4:])["id"])
    if uid not in ALLOWED:  # fail closed when ALLOWED_USER_IDS is empty
        raise HTTPException(403, "This Telegram account is not allowed to use this workspace")
    return uid


# ---------- integrations ----------
def integration_states(uid):
    out = []
    for iid, label, env in INTEGRATIONS:
        if kv_get(uid, f"disconnected:{iid}", False):
            state, detail = "disconnected", "Disconnected by you. Reconnect on the server."
        elif os.environ.get(env):
            state, detail = "configured", "Credentials present. Live connection check is not implemented."
        else:
            state, detail = "missing", f"Set {env} on the server."
        out.append({"id": iid, "label": label, "state": state, "detail": detail})
    if os.environ.get("PUBLISHER_ENABLED") == "1":
        out.append({"id": "publisher", "label": "Publishing worker", "state": "configured", "detail": "Enabled."})
    else:
        out.append({"id": "publisher", "label": "Publishing worker", "state": "not_implemented",
                    "detail": "No publishing worker is running. Set PUBLISHER_ENABLED=1 once it exists."})
    return out


def blockers_for(uid, level):
    st = {i["id"]: i for i in integration_states(uid)}
    return [{"id": r, "label": st[r]["label"], "state": st[r]["state"], "detail": st[r]["detail"]}
            for r in REQUIRED[level] if st[r]["state"] != "configured"]


def tiktok_ok(uid):
    return next(i for i in integration_states(uid) if i["id"] == "tiktok")["state"] == "configured"


# ---------- models ----------
class Permissions(BaseModel):
    publish: bool = False
    spend_budget: bool = False
    edit_queue: bool = False
    reply_comments: bool = False


class AgentSettings(BaseModel):
    niche: str = Field("", max_length=200)
    goals: str = Field("", max_length=1000)
    posts_per_week: int = Field(3, ge=0, le=50)
    budget_month: float = Field(0, ge=0, le=100000)
    language: str = Field("en", max_length=10)
    style: str = Field("", max_length=500)
    permissions: Permissions = Permissions()


class StatusIn(BaseModel):
    status: Literal["online", "paused"]


class AutonomyIn(BaseModel):
    level: Literal["planning", "production", "autonomous"]


class NotificationsIn(BaseModel):
    published: bool = True
    failed: bool = True
    budget_alert: bool = True
    weekly_report: bool = False


def vrow(r, full=False):
    d = {"id": r["id"], "title": r["title"], "status": r["status"], "created_at": r["created_at"],
         "scheduled_at": r["scheduled_at"], "published_at": r["published_at"],
         "cover_url": r["cover_url"], "metrics": json.loads(r["metrics"]) if r["metrics"] else None}
    if full:
        d.update(script=r["script"], caption=r["caption"])
    return d


# ---------- routes ----------
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/agent/status")
def agent_status(uid: int = Depends(current_user)):
    return {"status": kv_get(uid, "status", "paused"), "task": kv_get(uid, "task"),
            "autonomy": kv_get(uid, "autonomy", "planning")}


@app.get("/api/dashboard")
def dashboard(period: int = Query(7, ge=1, le=90), uid: int = Depends(current_user)):
    since = time.strftime("%Y-%m-%d", time.gmtime(time.time() - period * 86400))
    connected = tiktok_ok(uid)
    with db() as c:
        published = c.execute("SELECT COUNT(*) n FROM videos WHERE user_id=? AND status='published' AND published_at>=?",
                              (uid, since)).fetchone()["n"]
        d = c.execute("SELECT SUM(views) v FROM daily WHERE user_id=? AND date>=?", (uid, since)).fetchone()["v"]
        f = c.execute("SELECT followers FROM daily WHERE user_id=? ORDER BY date DESC LIMIT 1", (uid,)).fetchone()
        queue = [vrow(r) for r in c.execute(
            "SELECT * FROM videos WHERE user_id=? AND status IN ('scheduled','rendering') ORDER BY scheduled_at LIMIT 10", (uid,))]
        recent = [vrow(r) for r in c.execute(
            "SELECT * FROM videos WHERE user_id=? AND status='published' ORDER BY published_at DESC LIMIT 6", (uid,))]
        actions = [dict(r) for r in c.execute(
            "SELECT ts,text FROM actions WHERE user_id=? ORDER BY id DESC LIMIT 15", (uid,))]
    return {
        "agent": agent_status(uid),
        "tiktok": {"connected": connected, "account": kv_get(uid, "tiktok_account")},
        "stats": {"published": published, "views": d if connected else None,
                  "followers": f["followers"] if (f and connected) else None, "available": connected},
        "queue": queue, "recent": recent, "actions": actions,
    }


@app.get("/api/videos")
def videos(status: Optional[str] = None, uid: int = Depends(current_user)):
    q, args = "SELECT * FROM videos WHERE user_id=?", [uid]
    if status and status != "all":
        q += " AND status=?"; args.append(status)
    with db() as c:
        return {"items": [vrow(r) for r in c.execute(q + " ORDER BY id DESC LIMIT 100", args)]}


@app.get("/api/videos/{vid}")
def video(vid: int, uid: int = Depends(current_user)):
    with db() as c:
        r = c.execute("SELECT * FROM videos WHERE id=? AND user_id=?", (vid, uid)).fetchone()
        if not r:
            raise HTTPException(404, "Video not found")
        ev = [dict(e) for e in c.execute("SELECT ts,text FROM video_events WHERE video_id=? ORDER BY id", (vid,))]
    return {**vrow(r, True), "history": ev}


@app.get("/api/agent")
def agent(uid: int = Depends(current_user)):
    with db() as c:
        plans = [dict(r) for r in c.execute("SELECT ts,text FROM plans WHERE user_id=? ORDER BY id DESC LIMIT 10", (uid,))]
        dec = [dict(r) for r in c.execute("SELECT ts,text FROM decisions WHERE user_id=? ORDER BY id DESC LIMIT 20", (uid,))]
    s = agent_status(uid)
    return {**s, "settings": AgentSettings(**kv_get(uid, "settings", {})).model_dump(),
            "plans": plans, "decisions": dec,
            "requirements": {k: blockers_for(uid, k) for k in REQUIRED}}


@app.put("/api/agent/settings")
def save_settings(body: AgentSettings, uid: int = Depends(current_user)):
    kv_set(uid, "settings", body.model_dump())
    log_action(uid, "Agent settings updated")
    return {"ok": True}


@app.post("/api/agent/status")
def set_status(body: StatusIn, uid: int = Depends(current_user)):
    if body.status == "online":
        b = blockers_for(uid, kv_get(uid, "autonomy", "planning"))
        if b:
            raise HTTPException(409, {"message": "Required integrations are not ready", "blockers": b})
    kv_set(uid, "status", body.status)
    log_action(uid, f"Agent {body.status}")
    return {"status": body.status}


@app.post("/api/agent/autonomy")
def set_autonomy(body: AutonomyIn, uid: int = Depends(current_user)):
    b = blockers_for(uid, body.level)
    if b:
        raise HTTPException(409, {"message": f"Cannot enable {body.level}: integrations not ready", "blockers": b})
    kv_set(uid, "autonomy", body.level)
    log_action(uid, f"Autonomy set to {body.level}")
    return {"autonomy": body.level}


@app.get("/api/analytics")
def analytics(period: int = Query(30, ge=7, le=90), uid: int = Depends(current_user)):
    if not tiktok_ok(uid):
        return {"available": False, "reason": "TikTok is not connected, so no statistics can be loaded."}
    since = time.strftime("%Y-%m-%d", time.gmtime(time.time() - period * 86400))
    with db() as c:
        daily = [dict(r) for r in c.execute("SELECT date,views,followers FROM daily WHERE user_id=? AND date>=? ORDER BY date", (uid, since))]
        vids = [vrow(r) for r in c.execute("SELECT * FROM videos WHERE user_id=? AND status='published' AND published_at>=? ORDER BY published_at DESC", (uid, since))]
        exp = [dict(r) for r in c.execute("SELECT ts,text,result FROM experiments WHERE user_id=? ORDER BY id DESC LIMIT 20", (uid,))]
        ins = [dict(r) for r in c.execute("SELECT ts,text FROM insights WHERE user_id=? ORDER BY id DESC LIMIT 10", (uid,))]
    return {"available": True, "reason": None, "daily": daily, "videos": vids, "experiments": exp, "insights": ins}


@app.get("/api/settings")
def get_settings(uid: int = Depends(current_user)):
    return {"integrations": integration_states(uid),
            "notifications": NotificationsIn(**kv_get(uid, "notifications", {})).model_dump()}


@app.put("/api/settings/notifications")
def save_notifications(body: NotificationsIn, uid: int = Depends(current_user)):
    kv_set(uid, "notifications", body.model_dump())
    return {"ok": True}


@app.post("/api/integrations/{iid}/disconnect")
def disconnect(iid: str, uid: int = Depends(current_user)):
    if iid not in {i[0] for i in INTEGRATIONS}:
        raise HTTPException(404, "Unknown integration")
    kv_set(uid, f"disconnected:{iid}", True)
    if iid == "tiktok":
        kv_set(uid, "tiktok_account", None)
    if blockers_for(uid, kv_get(uid, "autonomy", "planning")):
        kv_set(uid, "status", "paused")
    log_action(uid, f"Disconnected {iid}")
    return {"ok": True}


@app.get("/api/export")
def export(uid: int = Depends(current_user)):
    out = {}
    with db() as c:
        for t in ("videos", "actions", "decisions", "plans", "experiments", "insights", "daily"):
            out[t] = [dict(r) for r in c.execute(f"SELECT * FROM {t} WHERE user_id=?", (uid,))]
        out["settings"] = {r["k"]: json.loads(r["v"]) for r in c.execute("SELECT k,v FROM kv WHERE user_id=?", (uid,))}
    return out


@app.delete("/api/data")
def delete_data(uid: int = Depends(current_user)):
    with db() as c:
        c.execute("DELETE FROM video_events WHERE video_id IN (SELECT id FROM videos WHERE user_id=?)", (uid,))
        for t in ("videos", "actions", "decisions", "plans", "experiments", "insights", "daily", "kv"):
            c.execute(f"DELETE FROM {t} WHERE user_id=?", (uid,))
    return {"ok": True}

from fastapi.responses import FileResponse

BASE = Path(__file__).resolve().parent


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/logo.svg")
def logo():
    return FileResponse(BASE / "logo.svg")

