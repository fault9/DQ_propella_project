"""
app.py — FastAPI pairwise rating app for Swedish PDF quality study.
Start: uvicorn app:app --reload  (from propella-eyeball/ directory)
"""
import os
import math
import random
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Set HF_HOME before any HF imports
os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "hf_cache"))

import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer, BadSignature
import base64

SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "results.db"
PAIRS_PATH = SCRIPT_DIR / "pairs.parquet"
SCHEMA_PATH = SCRIPT_DIR / "schema.sql"
TEMPLATES_DIR = SCRIPT_DIR / "templates"
STATIC_DIR = SCRIPT_DIR / "static"

RATER_SECRET = os.environ.get("RATER_SECRET", "dev-secret-change-me")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")
COOKIE_NAME = "rater"

# Calibration positions (1-indexed per spec → convert to 0-indexed)
CAL_POSITIONS_1IDX = [1, 8, 15, 16, 23, 30, 31, 38, 45]
CAL_POSITIONS_0IDX = [p - 1 for p in CAL_POSITIONS_1IDX]  # [0,7,14,15,22,29,30,37,44]

serializer = URLSafeSerializer(RATER_SECRET)

# ---------------------------------------------------------------------------
# Global state loaded at startup
# ---------------------------------------------------------------------------
pairs_dict: dict = {}   # pair_id -> row dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pairs_dict
    # Apply schema
    _apply_schema()
    # Load pairs
    if PAIRS_PATH.exists():
        df = pd.read_parquet(PAIRS_PATH)
        for _, row in df.iterrows():
            pairs_dict[row["pair_id"]] = row.to_dict()
    else:
        print("WARNING: pairs.parquet not found — run generate_pairs.py first.")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _apply_schema():
    with open(SCHEMA_PATH) as f:
        sql = f.read()
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(sql)
    # Add break timestamp columns to existing DBs (ALTER TABLE ignores errors if already present)
    for col in ("break1_started_at", "break2_started_at"):
        try:
            conn.execute(f"ALTER TABLE raters ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------
def get_rater_name(request: Request) -> str | None:
    cookie_val = request.cookies.get(COOKIE_NAME)
    if not cookie_val:
        return None
    try:
        name = serializer.loads(cookie_val)
        return name
    except BadSignature:
        return None


def set_rater_cookie(response: Response, name: str):
    token = serializer.dumps(name)
    response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax")


def require_rater(request: Request) -> str:
    name = get_rater_name(request)
    if not name:
        raise HTTPException(status_code=302, headers={"Location": "/"})
    return name


# ---------------------------------------------------------------------------
# Duplicate slot assignment
# All 9 dup pairs sorted by pair_id, shuffled with seed 42
# Slot 0: indices [0,1,2,3,4,5]; Slot 1: [0,1,2,6,7,8]; Slot 2: [3,4,5,6,7,8]
# ---------------------------------------------------------------------------
def get_dup_pairs_for_slot(slot: int) -> list:
    dup_pairs = [pid for pid, row in pairs_dict.items() if row.get("is_duplicate") and row.get("pair_kind") == "real"]
    dup_pairs.sort()
    rng = random.Random(42)
    rng.shuffle(dup_pairs)
    slot_indices = {
        0: [0, 1, 2, 3, 4, 5],
        1: [0, 1, 2, 6, 7, 8],
        2: [3, 4, 5, 6, 7, 8],
    }
    return [dup_pairs[i] for i in slot_indices[slot] if i < len(dup_pairs)]


# ---------------------------------------------------------------------------
# Queue construction
# ---------------------------------------------------------------------------
def build_queue(rater_name: str, conn: sqlite3.Connection):
    cur = conn.cursor()

    # Count raters with queue_built=1 inside a transaction
    cur.execute("SELECT COUNT(*) FROM raters WHERE queue_built=1")
    slot = cur.fetchone()[0]

    if slot >= 3:
        return {"error": "This pilot is full (3 raters maximum)."}

    # Non-dup real pairs
    non_dup_pairs = sorted([
        pid for pid, row in pairs_dict.items()
        if not row.get("is_duplicate") and row.get("pair_kind") == "real"
    ])
    rng_nondup = random.Random(42)
    rng_nondup.shuffle(non_dup_pairs)
    my_non_dup = non_dup_pairs[slot * 30:(slot + 1) * 30]

    # Dup real pairs
    my_dup = get_dup_pairs_for_slot(slot)

    # Calibration pairs
    cal_pairs = sorted([
        pid for pid, row in pairs_dict.items()
        if row.get("pair_kind") in ("calibration_readability", "calibration_substance")
    ])
    cal_rng = random.Random(hash(rater_name) ^ 0xCAFE)
    cal_rng.shuffle(cal_pairs)

    # Build 45-slot queue
    queue = [None] * 45
    # Place calibration at 0-indexed positions
    for i, pos in enumerate(CAL_POSITIONS_0IDX):
        if i < len(cal_pairs):
            queue[pos] = cal_pairs[i]

    # Fill remaining with real pairs (36 total: 30 non-dup + 6 dup)
    real_all = my_non_dup + my_dup
    real_rng = random.Random(hash(rater_name))
    real_rng.shuffle(real_all)

    real_iter = iter(real_all)
    for pos in range(45):
        if queue[pos] is None:
            try:
                queue[pos] = next(real_iter)
            except StopIteration:
                return {"error": "Not enough real pairs to fill queue."}

    # Side assignment
    side_rng = random.Random(hash(rater_name) ^ 0xDEAD)

    # Insert assignments
    for pos, pair_id in enumerate(queue):
        if pair_id is None:
            continue
        side = side_rng.choice(["a_left", "a_right"])
        cur.execute(
            "INSERT OR IGNORE INTO pair_assignments (rater_name, pair_id, queue_position, side_assignment) VALUES (?,?,?,?)",
            (rater_name, pair_id, pos, side),
        )

    # Mark queue built
    cur.execute("UPDATE raters SET queue_built=1 WHERE name=?", (rater_name,))
    conn.commit()

    return {"slot": slot}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def landing_get(request: Request):
    name = get_rater_name(request)
    if name:
        conn = get_db()
        row = conn.execute("SELECT queue_built FROM raters WHERE name=?", (name,)).fetchone()
        conn.close()
        if row and row["queue_built"] == 1:
            return RedirectResponse("/pair", status_code=302)
        return RedirectResponse("/rubric", status_code=302)
    return templates.TemplateResponse(request, "landing.html")


@app.post("/", response_class=HTMLResponse)
async def landing_post(request: Request, name: str = Form(...)):
    name = name.strip()
    if not name:
        return templates.TemplateResponse(request, "landing.html", {"error": "Please enter your name."})

    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO raters (name, first_seen_at) VALUES (?,?) ON CONFLICT(name) DO UPDATE SET first_seen_at=first_seen_at",
        (name, now),
    )
    conn.commit()
    conn.close()

    response = RedirectResponse("/rubric", status_code=302)
    set_rater_cookie(response, name)
    return response


@app.get("/rubric", response_class=HTMLResponse)
async def rubric_get(request: Request):
    name = get_rater_name(request)
    if not name:
        return RedirectResponse("/", status_code=302)

    conn = get_db()
    row = conn.execute("SELECT queue_built FROM raters WHERE name=?", (name,)).fetchone()
    conn.close()

    if row and row["queue_built"] == 1:
        return RedirectResponse("/pair", status_code=302)

    return templates.TemplateResponse(request, "rubric.html", {"rater_name": name})


@app.post("/rubric", response_class=HTMLResponse)
async def rubric_post(request: Request):
    name = get_rater_name(request)
    if not name:
        return RedirectResponse("/", status_code=302)

    form = await request.form()
    if not form.get("ack"):
        return templates.TemplateResponse(
            request, "rubric.html",
            {"rater_name": name, "error": "Please check the acknowledgement box."},
        )

    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()

    result = build_queue(name, conn)

    if "error" in result:
        conn.close()
        return templates.TemplateResponse(
            request, "rubric.html",
            {"rater_name": name, "error": result["error"]},
        )

    conn.execute("UPDATE raters SET rubric_ack_at=? WHERE name=?", (now, name))
    conn.commit()
    conn.close()

    return RedirectResponse("/pair", status_code=302)


@app.get("/pair", response_class=HTMLResponse)
async def pair_get(request: Request, after_break: int = 0):
    name = get_rater_name(request)
    if not name:
        return RedirectResponse("/", status_code=302)

    conn = get_db()
    row = conn.execute("SELECT queue_built FROM raters WHERE name=?", (name,)).fetchone()
    if not row or row["queue_built"] != 1:
        conn.close()
        return RedirectResponse("/rubric", status_code=302)

    completed = conn.execute(
        "SELECT COUNT(*) FROM responses WHERE rater_name=?", (name,)
    ).fetchone()[0]

    if completed >= 45:
        conn.close()
        return RedirectResponse("/done", status_code=302)

    if not after_break and completed in (15, 30):
        conn.close()
        return RedirectResponse("/break", status_code=302)

    next_row = conn.execute(
        """SELECT pa.pair_id, pa.queue_position, pa.side_assignment
           FROM pair_assignments pa
           WHERE pa.rater_name=?
             AND pa.pair_id NOT IN (SELECT pair_id FROM responses WHERE rater_name=?)
           ORDER BY pa.queue_position
           LIMIT 1""",
        (name, name),
    ).fetchone()
    conn.close()

    if not next_row:
        return RedirectResponse("/done", status_code=302)

    pair_id = next_row["pair_id"]
    queue_position = next_row["queue_position"]  # 0-indexed
    side_assignment = next_row["side_assignment"]

    pair = pairs_dict.get(pair_id)
    if not pair:
        raise HTTPException(status_code=500, detail="Pair data not found")

    # Resolve display order
    if side_assignment == "a_left":
        display_a_text = pair.get("doc_a_text") or ""
        display_b_text = pair.get("doc_b_text") or ""
        display_a_desc = pair.get("doc_a_description") or ""
        display_b_desc = pair.get("doc_b_description") or ""
        display_a_char_count = pair.get("doc_a_char_count", 0)
        display_b_char_count = pair.get("doc_b_char_count", 0)
    else:  # a_right: doc_b shown as A (left)
        display_a_text = pair.get("doc_b_text") or ""
        display_b_text = pair.get("doc_a_text") or ""
        display_a_desc = pair.get("doc_b_description") or ""
        display_b_desc = pair.get("doc_a_description") or ""
        display_a_char_count = pair.get("doc_b_char_count", 0)
        display_b_char_count = pair.get("doc_a_char_count", 0)

    # 1-indexed queue position for display
    queue_pos_1idx = queue_position + 1
    session_num = math.ceil(queue_pos_1idx / 15)
    completed_in_session = ((queue_pos_1idx - 1) % 15) + 1

    return templates.TemplateResponse(request, "pair.html", {
        "pair_id": pair_id,
        "side_assignment": side_assignment,
        "queue_position": queue_pos_1idx,
        "session_num": session_num,
        "completed_in_session": completed_in_session,
        "display_a_text": display_a_text,
        "display_b_text": display_b_text,
        "display_a_desc": display_a_desc,
        "display_b_desc": display_b_desc,
        "display_a_char_count": display_a_char_count,
        "display_b_char_count": display_b_char_count,
        "rater_name": name,
    })


@app.post("/response", response_class=HTMLResponse)
async def response_post(request: Request):
    name = get_rater_name(request)
    if not name:
        return RedirectResponse("/", status_code=302)

    form = await request.form()
    pair_id = form.get("pair_id", "").strip()
    readability_choice = form.get("readability_choice", "").strip() or None
    substance_choice = form.get("substance_choice", "").strip() or None
    notes = form.get("notes", "").strip() or None
    time_ms_str = form.get("time_ms", "")
    skipped = 1 if form.get("skipped") else 0

    try:
        time_ms = int(time_ms_str) if time_ms_str else None
    except ValueError:
        time_ms = None

    # Validate
    if not skipped:
        if not readability_choice or not substance_choice:
            # Re-render pair with error — just redirect for simplicity
            return RedirectResponse("/pair?after_break=1", status_code=302)

    pair = pairs_dict.get(pair_id)
    if not pair:
        return RedirectResponse("/pair?after_break=1", status_code=302)

    conn = get_db()
    pa_row = conn.execute(
        "SELECT queue_position, side_assignment FROM pair_assignments WHERE rater_name=? AND pair_id=?",
        (name, pair_id),
    ).fetchone()

    if not pa_row:
        conn.close()
        return RedirectResponse("/pair?after_break=1", status_code=302)

    queue_position = pa_row["queue_position"]
    side_assignment = pa_row["side_assignment"]
    session_num = math.ceil((queue_position + 1) / 15)
    pair_kind = pair.get("pair_kind", "real")
    is_dup = 1 if pair.get("is_duplicate") else 0

    response_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """INSERT OR IGNORE INTO responses
           (response_id, rater_name, pair_id, pair_kind, is_duplicate, queue_position, session_num,
            side_assignment, readability_choice, substance_choice, skipped, notes, time_ms, submitted_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (response_id, name, pair_id, pair_kind, is_dup, queue_position, session_num,
         side_assignment, readability_choice, substance_choice, skipped, notes, time_ms, now),
    )
    conn.commit()

    new_count = conn.execute(
        "SELECT COUNT(*) FROM responses WHERE rater_name=?", (name,)
    ).fetchone()[0]
    conn.close()

    if new_count >= 45:
        return RedirectResponse("/done", status_code=302)
    if new_count in (15, 30):
        return RedirectResponse("/break", status_code=302)
    return RedirectResponse("/pair?after_break=1", status_code=302)


@app.get("/break", response_class=HTMLResponse)
async def break_get(request: Request):
    name = get_rater_name(request)
    if not name:
        return RedirectResponse("/", status_code=302)

    conn = get_db()
    completed = conn.execute(
        "SELECT COUNT(*) FROM responses WHERE rater_name=?", (name,)
    ).fetchone()[0]

    if completed >= 30:
        session_just_completed = 2
        session_next = 3
        col = "break2_started_at"
    else:
        session_just_completed = 1
        session_next = 2
        col = "break1_started_at"

    row = conn.execute(f"SELECT {col} FROM raters WHERE name=?", (name,)).fetchone()
    now = datetime.now(timezone.utc)
    started_at = row[col] if row else None

    if not started_at:
        # First visit — record timestamp
        conn.execute(f"UPDATE raters SET {col}=? WHERE name=?",
                     (now.isoformat(), name))
        conn.commit()
        seconds_remaining = 600
    else:
        elapsed = (now - datetime.fromisoformat(started_at)).total_seconds()
        seconds_remaining = max(0, int(600 - elapsed))

    conn.close()

    return templates.TemplateResponse(request, "break.html", {
        "session_just_completed": session_just_completed,
        "session_next": session_next,
        "completed": completed,
        "seconds_remaining": seconds_remaining,
    })


@app.get("/done", response_class=HTMLResponse)
async def done_get(request: Request):
    name = get_rater_name(request)
    return templates.TemplateResponse(request, "done.html", {"rater_name": name})


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

def check_admin_auth(request: Request) -> bool:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth[6:]).decode("utf-8")
        username, password = decoded.split(":", 1)
        return username == "admin" and password == ADMIN_PASSWORD
    except Exception:
        return False


def chosen_doc(choice: str, side_assignment: str) -> str:
    """Return original doc 'a' or 'b' based on display choice and side assignment."""
    chose_a_display = choice in ("a_much", "a_slight")
    if side_assignment == "a_left":
        return "a" if chose_a_display else "b"
    else:  # a_right
        return "b" if chose_a_display else "a"


@app.get("/admin", response_class=HTMLResponse)
async def admin_get(request: Request):
    if not check_admin_auth(request):
        return Response(
            content="Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Admin"'},
        )

    conn = get_db()

    raters = conn.execute("SELECT name, first_seen_at, rubric_ack_at, queue_built FROM raters").fetchall()

    rater_stats = []
    for rater_row in raters:
        rn = rater_row["name"]
        responses = conn.execute(
            "SELECT * FROM responses WHERE rater_name=?", (rn,)
        ).fetchall()

        total = len(responses)
        skipped_count = sum(1 for r in responses if r["skipped"])
        completed_count = total - skipped_count

        real_count = sum(1 for r in responses if r["pair_kind"] == "real" and not r["is_duplicate"] and not r["skipped"])
        dup_count = sum(1 for r in responses if r["is_duplicate"] and not r["skipped"])
        cal_read_count = sum(1 for r in responses if r["pair_kind"] == "calibration_readability" and not r["skipped"])
        cal_sub_count = sum(1 for r in responses if r["pair_kind"] == "calibration_substance" and not r["skipped"])

        times = [r["time_ms"] for r in responses if not r["skipped"] and r["time_ms"] is not None]
        mean_time = sum(times) / len(times) if times else None

        # Calibration accuracy
        def cal_accuracy(kind):
            correct = 0
            total_cal = 0
            for r in responses:
                if r["pair_kind"] != kind or r["skipped"]:
                    continue
                pair = pairs_dict.get(r["pair_id"])
                if not pair:
                    continue
                correct_doc = pair.get("calibration_correct_doc")
                if not correct_doc or not r["readability_choice"] and not r["substance_choice"]:
                    continue
                if kind == "calibration_readability":
                    choice = r["readability_choice"]
                else:
                    choice = r["substance_choice"]
                if not choice:
                    continue
                picked = chosen_doc(choice, r["side_assignment"])
                total_cal += 1
                if picked == correct_doc:
                    correct += 1
            return f"{correct}/{total_cal}" if total_cal > 0 else "n/a"

        cal_read_acc = cal_accuracy("calibration_readability")
        cal_sub_acc = cal_accuracy("calibration_substance")

        # Choice distributions
        read_choices = {}
        sub_choices = {}
        for r in responses:
            if r["skipped"]:
                continue
            rc = r["readability_choice"] or "null"
            sc = r["substance_choice"] or "null"
            read_choices[rc] = read_choices.get(rc, 0) + 1
            sub_choices[sc] = sub_choices.get(sc, 0) + 1

        # Divergence
        diverge_count = 0
        diverge_total = 0
        for r in responses:
            if r["skipped"] or not r["readability_choice"] or not r["substance_choice"]:
                continue
            r_dir = "a" if r["readability_choice"] in ("a_much", "a_slight") else "b"
            s_dir = "a" if r["substance_choice"] in ("a_much", "a_slight") else "b"
            diverge_total += 1
            if r_dir != s_dir:
                diverge_count += 1
        diverge_pct = f"{100*diverge_count/diverge_total:.1f}%" if diverge_total > 0 else "n/a"

        rater_stats.append({
            "name": rn,
            "first_seen_at": rater_row["first_seen_at"],
            "queue_built": rater_row["queue_built"],
            "total_responses": total,
            "skipped": skipped_count,
            "real_count": real_count,
            "dup_count": dup_count,
            "cal_read_count": cal_read_count,
            "cal_sub_count": cal_sub_count,
            "mean_time_ms": f"{mean_time:.0f}" if mean_time else "n/a",
            "cal_read_acc": cal_read_acc,
            "cal_sub_acc": cal_sub_acc,
            "read_choices": read_choices,
            "sub_choices": sub_choices,
            "diverge_pct": diverge_pct,
        })

    # Cross-rater duplicate agreement
    dup_pairs_ids = sorted([pid for pid, row in pairs_dict.items() if row.get("is_duplicate") and row.get("pair_kind") == "real"])
    dup_agreement = []
    exact_read_agree = 0
    exact_sub_agree = 0
    binary_read_agree = 0
    binary_sub_agree = 0
    dup_pair_total = 0

    for dp_id in dup_pairs_ids:
        responses_for_pair = conn.execute(
            "SELECT * FROM responses WHERE pair_id=?", (dp_id,)
        ).fetchall()
        entry = {"pair_id": dp_id, "ratings": []}
        for r in responses_for_pair:
            entry["ratings"].append({
                "rater": r["rater_name"],
                "readability_choice": r["readability_choice"],
                "substance_choice": r["substance_choice"],
                "skipped": r["skipped"],
            })
        dup_agreement.append(entry)

    # Pairwise agreement across raters for each dup pair
    for entry in dup_agreement:
        non_skip = [r for r in entry["ratings"] if not r["skipped"]]
        for i in range(len(non_skip)):
            for j in range(i + 1, len(non_skip)):
                r1, r2 = non_skip[i], non_skip[j]
                if r1["readability_choice"] and r2["readability_choice"]:
                    dup_pair_total += 1
                    if r1["readability_choice"] == r2["readability_choice"]:
                        exact_read_agree += 1
                    r1_bin = "a" if r1["readability_choice"] in ("a_much", "a_slight") else "b"
                    r2_bin = "a" if r2["readability_choice"] in ("a_much", "a_slight") else "b"
                    if r1_bin == r2_bin:
                        binary_read_agree += 1
                if r1["substance_choice"] and r2["substance_choice"]:
                    if r1["substance_choice"] == r2["substance_choice"]:
                        exact_sub_agree += 1
                    r1_bin = "a" if r1["substance_choice"] in ("a_much", "a_slight") else "b"
                    r2_bin = "a" if r2["substance_choice"] in ("a_much", "a_slight") else "b"
                    if r1_bin == r2_bin:
                        binary_sub_agree += 1

    exact_read_pct = f"{100*exact_read_agree/dup_pair_total:.1f}%" if dup_pair_total > 0 else "n/a"
    exact_sub_pct = f"{100*exact_sub_agree/dup_pair_total:.1f}%" if dup_pair_total > 0 else "n/a"
    binary_read_pct = f"{100*binary_read_agree/dup_pair_total:.1f}%" if dup_pair_total > 0 else "n/a"
    binary_sub_pct = f"{100*binary_sub_agree/dup_pair_total:.1f}%" if dup_pair_total > 0 else "n/a"

    conn.close()

    return templates.TemplateResponse(request, "admin.html", {
        "rater_stats": rater_stats,
        "dup_agreement": dup_agreement,
        "exact_read_pct": exact_read_pct,
        "exact_sub_pct": exact_sub_pct,
        "binary_read_pct": binary_read_pct,
        "binary_sub_pct": binary_sub_pct,
        "total_pairs": len(pairs_dict),
    })
