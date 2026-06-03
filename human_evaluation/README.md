# Swedish PDF Quality Study — Pairwise Rating Tool

A local pairwise A/B rating tool for evaluating Swedish PDF document quality.

## Prerequisites

Python 3.8+ is required. Install dependencies:

```bash
pip install fastapi "uvicorn[standard]" jinja2 python-multipart itsdangerous pandas pyarrow
```

Or using the project venv:

```bash
.venv/bin/pip install fastapi "uvicorn[standard]" jinja2 python-multipart itsdangerous pandas pyarrow
```

## Step 1: Generate pairs

Run this once before starting the server. It reads `sample.parquet` and writes `pairs.parquet`.

```bash
python generate_pairs.py
# or with the venv:
.venv/bin/python3 generate_pairs.py
```

The script is idempotent: if `pairs.parquet` already exists, it prints a message and exits without changes. Delete `pairs.parquet` to regenerate.

## Step 2: Start the server

From the `propella-eyeball/` directory:

```bash
uvicorn app:app --reload --host 0.0.0.0
# or with the venv:
.venv/bin/uvicorn app:app --reload --host 0.0.0.0
```

The app will be available at http://localhost:8000 on your own machine.

## Step 3: Share with remote raters (ngrok)

If raters are on different networks or the local IP is unreachable (common on university WiFi), use ngrok to create a public tunnel.

**One-time setup:**
1. Create a free account at https://dashboard.ngrok.com/signup
2. Copy your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
3. Run: `ngrok config add-authtoken YOUR_TOKEN_HERE`

**Each session** — in a second terminal while uvicorn is running:
```bash
ngrok http 8000
```

Ngrok prints a public URL like `https://xxxx.ngrok-free.dev` — send that to all raters. The URL changes each time you restart ngrok, so don't restart it mid-session.

Both terminals (uvicorn + ngrok) must stay open for the duration of the study.

**Same-network alternative:** If all raters are on the same WiFi, find your local IP with `ipconfig getifaddr en0` and share `http://<your-ip>:8000` instead. No ngrok needed.

## Usage

- Each rater opens the shared link, enters their name, reads the rubric, then completes 3 sessions of 15 pairs each.
- The study supports up to 3 raters.
- Calibration pairs (9 per rater) are interspersed automatically.
- All responses are stored in `results.db` on the machine running uvicorn.

## Admin panel

View progress and inter-rater agreement at http://localhost:8000/admin

- Username: `admin`
- Password: set the `ADMIN_PASSWORD` environment variable (default: `admin`)

```bash
ADMIN_PASSWORD=mysecret uvicorn app:app --reload
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `RATER_SECRET` | `dev-secret-change-me` | Cookie signing key — change for production |
| `ADMIN_PASSWORD` | `admin` | HTTP Basic auth password for /admin |

## Backing up results

```bash
cp results.db results_backup_$(date +%Y%m%d).db
```

## Exporting results to CSV

```bash
sqlite3 -header -csv results.db "SELECT * FROM responses;" > responses.csv
```

## Cleaning up (reset everything)

```bash
rm -rf hf_cache/ results.db pairs.parquet
```

Then re-run `generate_pairs.py` before restarting the server.

## File overview

| File | Purpose |
|---|---|
| `generate_pairs.py` | Generates `pairs.parquet` from `sample.parquet` |
| `app.py` | FastAPI web application |
| `schema.sql` | SQLite schema for `results.db` |
| `templates/` | Jinja2 HTML templates |
| `static/style.css` | CSS styles |
| `pairs.parquet` | Generated pair assignments (after running generate_pairs.py) |
| `results.db` | Rater responses (created on first server start) |
