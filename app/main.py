import io
import json
import sqlite3
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import agents, redflags
from .schemas import Intake

app = FastAPI(title="Shifa Intake")
CASES: dict[str, dict] = {}  # in-memory cache
ROOT = Path(__file__).resolve().parent.parent

DB_PATH = Path(tempfile.gettempdir()) / "shifa_cases_cache.db"

def _db_init():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, payload TEXT)")
    except Exception:
        pass

_db_init()

def _save_case(cid: str, data: dict):
    CASES[cid] = data
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT OR REPLACE INTO cases (id, payload) VALUES (?, ?)", (cid, json.dumps(data)))
    except Exception:
        pass

def _get_case(cid: str) -> dict | None:
    if cid in CASES:
        return CASES[cid]
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute("SELECT payload FROM cases WHERE id = ?", (cid,))
            row = cur.fetchone()
            if row:
                d = json.loads(row[0])
                CASES[cid] = d
                return d
    except Exception:
        pass
    return None


class IntakeIn(BaseModel):
    text: str


class ApproveIn(BaseModel):
    intake: Intake
    reviewer: str = "staff"


class InstructionsIn(BaseModel):
    english: str


class TTSIn(BaseModel):
    text: str


@app.post("/api/intake")
def create_case(body: IntakeIn):
    if not body.text.strip():
        raise HTTPException(400, "Empty text")
    flags = redflags.check(body.text)  # rule-based, independent of the model
    try:
        intake = agents.intake_agent(body.text)
        summary, blocked = agents.summary_agent(intake)
    except Exception as e:
        raise HTTPException(502, f"Model error, please send this case to manual review: {e}")
    cid = uuid.uuid4().hex[:8]
    case_data = {"id": cid, "text": body.text, "intake": intake.model_dump(), "red_flags": flags,
                 "summary": summary, "guardrail_blocked": blocked, "status": "pending_staff_review",
                 "draft_urdu": None, "draft_urdu_script": None, "instructions_en": None, "suggested_instructions": ""}
    _save_case(cid, case_data)
    return case_data


@app.post("/api/cases/{cid}/approve")
def approve(cid: str, body: ApproveIn):
    c = _get_case(cid) or _404()
    c["intake"] = body.intake.model_dump()  # staff may have corrected fields
    c["summary"], c["guardrail_blocked"] = agents.summary_agent(body.intake)
    c["suggested_instructions"] = agents.suggest_instructions_agent(body.intake)
    c["status"], c["reviewer"] = "intake_approved", body.reviewer
    _save_case(cid, c)
    return c


@app.post("/api/cases/{cid}/instructions")
def instructions(cid: str, body: InstructionsIn):
    c = _get_case(cid) or _404()
    if c["status"] != "intake_approved":
        raise HTTPException(409, "Approve the intake first")
    c["instructions_en"] = body.english
    c["draft_urdu"] = agents.translate_agent(body.english)
    try:
        c["draft_urdu_script"] = agents.roman_to_urdu_script(c["draft_urdu"])
    except Exception:
        c["draft_urdu_script"] = ""
    c["status"] = "translation_pending_review"
    _save_case(cid, c)
    return c


@app.post("/api/cases/{cid}/release")
def release(cid: str):
    c = _get_case(cid) or _404()
    if c["status"] != "translation_pending_review":
        raise HTTPException(409, "Nothing to release")
    c["status"] = "released_to_patient"  # only after a human approved the translation
    _save_case(cid, c)
    return c


@app.post("/api/tts")
def text_to_speech(body: TTSIn):
    if not body.text.strip():
        raise HTTPException(400, "Empty text")
    try:
        audio_bytes = agents.generate_urdu_tts(body.text)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=advice.mp3"}
        )
    except Exception as e:
        raise HTTPException(500, f"TTS generation failed: {e}")


@app.post("/api/transcribe")
async def transcribe_voice(audio: UploadFile = File(...)):
    try:
        content = await audio.read()
        if not content:
            raise HTTPException(400, "Empty audio file")
        text = agents.transcribe_audio_agent(content, audio.filename or "audio.webm")
        return {"text": text}
    except Exception as e:
        raise HTTPException(500, f"Audio transcription failed: {e}")


@app.get("/api/health")
def health():
    return {"ok": True}


def _404():
    raise HTTPException(404, "Case not found")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/presentation")
def presentation():
    return FileResponse(ROOT / "frontend" / "presentation.html")
