"""
eyeball.py — Swedish PDF training data inspection script.

Downloads propella-1 annotations + finepdfs text, joins them,
writes sample.parquet and inspect.html, then prints cleanup info.

Cache isolation: HF_HOME is set to ./hf_cache BEFORE any HF imports.
"""

import os
import sys

# ── 0. Cache isolation (must happen before any HF import) ──────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HF_CACHE   = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE          # affects all subsequent HF activity
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")

# ── Now safe to import HF libs ─────────────────────────────────────────────
import html
import math
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset

SAMPLE_PARQUET = os.path.join(SCRIPT_DIR, "sample.parquet")
INSPECT_HTML   = os.path.join(SCRIPT_DIR, "inspect.html")

SEED            = 0
ANN_REPO        = "openeurollm/propella-annotations"
ANN_CONFIG      = "finepdfs"
ANN_SPLIT       = "swe_Latn"
# Direct path to the single Swedish shard — avoids downloading all language shards
ANN_SWE_FILE    = "data/propella-1-4b/finepdfs/swe_Latn/shard000000.parquet"
TEXT_REPO       = "HuggingFaceFW/finepdfs"
TEXT_CONFIG     = "swe_Latn"
TEXT_SPLIT      = "train"

STRAT_FLOOR     = 5
STRAT_CAP       = 25
STREAM_SAFETY   = 5_000_000   # finepdfs swe_Latn train has ~4.1M rows
STREAM_TOTAL    = 4_125_553   # known row count, used for progress %
PROGRESS_EVERY  = 5_000
DISPLAY_TRUNC   = 5000

random.seed(SEED)


# ── helpers ────────────────────────────────────────────────────────────────

def dir_size_bytes(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def file_size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def pct_latin(text: str) -> float:
    if not text:
        return 0.0
    latin = sum(1 for c in text if unicodedata.category(c).startswith("L")
                and unicodedata.name(c, "").startswith(("LATIN", "latin")))
    letters = sum(1 for c in text if unicodedata.category(c).startswith("L"))
    return latin / letters * 100 if letters else 0.0


def coerce_str(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v) if v is not None else ""


def normalize_id(v: str) -> str:
    """Strip <urn:uuid:...> wrapper so bare and wrapped UUIDs match."""
    if v and v.startswith("<urn:uuid:") and v.endswith(">"):
        return v[10:-1]
    return v


# ── 1. Load annotation side ────────────────────────────────────────────────
# Use data_files to fetch only the swe_Latn shard — avoids pulling all languages
print("\n=== Loading annotations (non-streaming, swe_Latn shard only) ===")
ann_ds = load_dataset(
    ANN_REPO,
    data_files=ANN_SWE_FILE,
    split="train",        # data_files loads always land in "train"
)
print(f"  Annotation rows: {len(ann_ds):,}")
print(f"  Columns: {ann_ds.column_names}")

# Verify id field exists and show 5 examples
assert "id" in ann_ds.column_names, "No 'id' column in annotations!"
ann_ids_sample = [ann_ds[i]["id"] for i in range(min(5, len(ann_ds)))]
print(f"  First 5 annotation ids: {ann_ids_sample}")


# ── 2. Verify text side (first 100 rows) ──────────────────────────────────
print("\n=== Verifying text side (first 100 rows, streaming) ===")
text_ds_stream = load_dataset(TEXT_REPO, TEXT_CONFIG, split=TEXT_SPLIT, streaming=True)
text_head = []
for row in text_ds_stream:
    text_head.append(row)
    if len(text_head) >= 100:
        break

assert text_head, "No rows returned from finepdfs!"
text_cols = list(text_head[0].keys())
print(f"  finepdfs columns (first row): {text_cols}")
assert "id" in text_cols, "No 'id' column in finepdfs!"

text_ids_sample = [r["id"] for r in text_head[:5]]
print(f"  First 5 finepdfs ids: {text_ids_sample}")

# Format check: both should look like the same type
ann_id_type  = type(ann_ds[0]["id"]).__name__
text_id_type = type(text_head[0]["id"]).__name__
print(f"  id type — annotations: {ann_id_type}, finepdfs: {text_id_type}")

if ann_id_type != text_id_type:
    print(f"\n[STOP] id type mismatch: annotations={ann_id_type}, finepdfs={text_id_type}")
    print("Cannot safely join. Exiting.")
    sys.exit(1)

# Quick overlap check on first 100 (normalize both sides)
ann_first100_ids  = set(normalize_id(ann_ds[i]["id"]) for i in range(min(100, len(ann_ds))))
text_first100_ids = set(normalize_id(r["id"]) for r in text_head)
overlap = ann_first100_ids & text_first100_ids
print(f"  Overlap in first-100 ids: {len(overlap)} / 100 (ann) x 100 (text)")
if len(overlap) == 0:
    print("\n[STOP] Zero overlap in first-100 id check. "
          "The datasets may not join on 'id'. Exiting.")
    sys.exit(1)
print("  Join verification: OK")


# ── 3. Stratified sample ───────────────────────────────────────────────────
if os.path.exists(SAMPLE_PARQUET):
    print(f"\n=== sample.parquet exists — loading cached join ===")
    df = pd.read_parquet(SAMPLE_PARQUET)
    print(f"  Loaded {len(df):,} rows from {SAMPLE_PARQUET}")
else:
    print("\n=== Stratified sampling on content_quality × educational_value ===")

    STRAT_COLS = ["content_quality", "educational_value"]
    for col in STRAT_COLS:
        if col not in ann_ds.column_names:
            print(f"[WARNING] Column '{col}' missing from annotations. "
                  "Falling back to sampling entire dataset.")

    ann_df = ann_ds.to_pandas()
    print(f"  Converted annotations to DataFrame: {len(ann_df):,} rows")

    # Build strata
    for col in STRAT_COLS:
        if col not in ann_df.columns:
            ann_df[col] = "unknown"
    ann_df["_stratum"] = (ann_df["content_quality"].astype(str)
                          + "×"
                          + ann_df["educational_value"].astype(str))

    strata_counts = ann_df["_stratum"].value_counts()
    print(f"  Unique strata: {len(strata_counts)}")
    print(f"  Stratum distribution (top 20):\n{strata_counts.head(20).to_string()}")

    # Sample per stratum
    sampled_ids = []
    sparse_cells = []
    empty_cells  = []

    for stratum, group in ann_df.groupby("_stratum"):
        n = len(group)
        if n == 0:
            empty_cells.append(stratum)
            continue
        target = max(STRAT_FLOOR, min(STRAT_CAP, round(n / len(ann_df) * 300)))
        target = min(target, n)
        if n < STRAT_FLOOR:
            sparse_cells.append((stratum, n))
            target = n  # take all
        sample = group.sample(n=target, random_state=SEED)
        sampled_ids.extend(sample["id"].tolist())

    print(f"\n  Total sampled ids: {len(sampled_ids)}")
    if sparse_cells:
        print(f"  Sparse cells (<{STRAT_FLOOR} docs): {sparse_cells}")
    if empty_cells:
        print(f"  Empty cells: {empty_cells}")

    # Normalize sampled ids and build lookup
    ann_df["_norm_id"] = ann_df["id"].map(normalize_id)
    sampled_id_set     = set(normalize_id(i) for i in sampled_ids)
    ann_sampled        = ann_df[ann_df["_norm_id"].isin(sampled_id_set)].copy()

    # ── 4. Stream finepdfs and collect matching texts ──────────────────────
    print(f"\n=== Streaming finepdfs to fetch {len(sampled_id_set)} texts ===")
    text_ds_stream2 = load_dataset(TEXT_REPO, TEXT_CONFIG, split=TEXT_SPLIT, streaming=True)

    found_texts = {}   # normalized id -> {text, url}
    remaining   = set(sampled_id_set)
    streamed    = 0

    for row in text_ds_stream2:
        streamed += 1
        rid = normalize_id(row.get("id", ""))
        if rid in remaining:
            found_texts[rid] = {
                "text": row.get("text", ""),
                "url":  row.get("url", ""),
            }
            remaining.discard(rid)
        if streamed % PROGRESS_EVERY == 0 or not remaining:
            pct = streamed / STREAM_TOTAL * 100
            print(f"  {streamed:>9,} / ~{STREAM_TOTAL:,} rows ({pct:5.1f}%) — "
                  f"found {len(found_texts)}/{len(sampled_id_set)}, "
                  f"still seeking {len(remaining)}")
        if not remaining or streamed >= STREAM_SAFETY:
            break

    print(f"  Finished streaming. Total streamed: {streamed:,}")
    if remaining:
        print(f"  [WARNING] {len(remaining)} ids not found in finepdfs: "
              f"{list(remaining)[:10]}{'...' if len(remaining) > 10 else ''}")
    else:
        print("  All sampled ids found in finepdfs.")

    # ── 5. Build joined DataFrame and persist ─────────────────────────────
    text_df = pd.DataFrame([
        {"_norm_id": rid, "text": v["text"], "url": v["url"]}
        for rid, v in found_texts.items()
    ])
    df = ann_sampled.merge(text_df, on="_norm_id", how="left")
    df.drop(columns=["_norm_id"], inplace=True, errors="ignore")
    df["char_count"] = df["text"].fillna("").str.len()
    df.drop(columns=["_stratum"], inplace=True, errors="ignore")

    df.to_parquet(SAMPLE_PARQUET, index=False)
    print(f"  Wrote {len(df):,} rows to {SAMPLE_PARQUET}")


# ── 6. Summary stats ───────────────────────────────────────────────────────
print("\n=== Summary statistics ===")

# Rebuild stratum column
for col in ["content_quality", "educational_value"]:
    if col not in df.columns:
        df[col] = "unknown"
df["_stratum"] = (df["content_quality"].astype(str)
                  + "×"
                  + df["educational_value"].astype(str))

null_texts = df["text"].isna().sum() + (df["text"] == "").sum()
print(f"  Empty/null text fields: {null_texts}")

strat_stats = (
    df.groupby("_stratum")["char_count"]
    .agg(count="count",
         median=lambda x: round(x.median()),
         p90=lambda x: round(x.quantile(0.9)))
    .reset_index()
    .sort_values("count", ascending=False)
)
print("\n  Stratum | count | median_chars | p90_chars")
print("  " + "-" * 55)
for _, r in strat_stats.iterrows():
    print(f"  {r['_stratum']:<35} | {r['count']:>5} | {r['median']:>12,} | {r['p90']:>9,}")

# % Latin-script sanity check (sample of up to 50 docs)
sample_for_latin = df["text"].dropna().head(50).tolist()
pct_latin_vals   = [pct_latin(t) for t in sample_for_latin if t]
avg_latin        = sum(pct_latin_vals) / len(pct_latin_vals) if pct_latin_vals else 0
print(f"\n  Avg % Latin-script letters (sample n={len(pct_latin_vals)}): {avg_latin:.1f}%")
if avg_latin < 70:
    print("  [WARNING] Low Latin-script %. May not be Swedish text.")


# ── 7. Generate inspect.html ───────────────────────────────────────────────
print(f"\n=== Generating {INSPECT_HTML} ===")

QUALITY_ORDER = sorted(df["content_quality"].unique(), reverse=True)

# Self-contained CSS + JS
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; font-size: 14px;
       background: #f5f5f5; color: #222; padding: 1rem; }
h1 { margin-bottom: 1rem; font-size: 1.4rem; }
.jump-links { margin-bottom: 1.5rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
.jump-links a { background: #0066cc; color: #fff; padding: 0.3rem 0.7rem;
                border-radius: 4px; text-decoration: none; font-size: 0.85rem; }
.jump-links a:hover { background: #004499; }
.stratum-section { margin-bottom: 2.5rem; }
.stratum-heading { font-size: 1.1rem; font-weight: bold; color: #444;
                   border-bottom: 2px solid #ccc; padding-bottom: 0.4rem;
                   margin-bottom: 1rem; }
details { background: #fff; border: 1px solid #ddd; border-radius: 6px;
          margin-bottom: 0.8rem; padding: 0.7rem 1rem; }
details[open] { border-color: #0066cc; }
summary { cursor: pointer; font-weight: 600; font-size: 0.95rem;
          list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "▶ "; font-size: 0.7rem; color: #888; }
details[open] summary::before { content: "▼ "; }
.ann-table { width: 100%; border-collapse: collapse; margin: 0.7rem 0; font-size: 0.82rem; }
.ann-table th { text-align: left; padding: 3px 8px; background: #f0f0f0;
                border-bottom: 1px solid #ddd; white-space: nowrap; }
.ann-table td { padding: 3px 8px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
.text-block { background: #fafafa; border: 1px solid #e5e5e5; border-radius: 4px;
              padding: 0.7rem; margin-top: 0.5rem; white-space: pre-wrap;
              font-family: monospace; font-size: 0.8rem; line-height: 1.5;
              max-height: 300px; overflow: hidden; }
.text-block.expanded { max-height: none; }
.toggle-btn { background: none; border: 1px solid #0066cc; color: #0066cc;
              border-radius: 4px; padding: 0.2rem 0.6rem; cursor: pointer;
              font-size: 0.8rem; margin-top: 0.4rem; }
.toggle-btn:hover { background: #e8f0fb; }
.url-link { font-size: 0.75rem; color: #888; word-break: break-all; }
"""

JS = """
function toggleText(btn, id) {
  var el = document.getElementById(id);
  if (el.classList.contains('expanded')) {
    el.classList.remove('expanded');
    btn.textContent = 'Show more';
  } else {
    el.classList.add('expanded');
    btn.textContent = 'Show less';
  }
}
"""

ANN_FIELDS = [c for c in df.columns
              if c not in ("id", "text", "url", "char_count", "_stratum",
                           "content_quality", "educational_value",
                           "one_sentence_description")]
# put quality axes first
DISPLAY_ANN = (["content_quality", "educational_value"]
               + [f for f in ANN_FIELDS])


def render_card(row, idx):
    desc = html.escape(coerce_str(row.get("one_sentence_description", "")) or "(no description)")
    url  = html.escape(str(row.get("url", "") or ""))
    text = str(row.get("text", "") or "")
    char_count = row.get("char_count", len(text))

    truncated   = text[:DISPLAY_TRUNC]
    need_toggle = len(text) > DISPLAY_TRUNC
    text_id     = f"txt_{idx}"

    # annotation table
    rows_html = ""
    for field in DISPLAY_ANN:
        val = coerce_str(row.get(field, ""))
        rows_html += (f"<tr><th>{html.escape(field)}</th>"
                      f"<td>{html.escape(val)}</td></tr>\n")

    toggle_html = ""
    if need_toggle:
        toggle_html = (f'<button class="toggle-btn" onclick="toggleText(this,\'{text_id}\')">'
                       f'Show more</button>')

    return f"""
<details>
  <summary>{desc} <span style="font-weight:normal;color:#888;font-size:0.8rem">[{char_count:,} chars]</span></summary>
  <table class="ann-table">{rows_html}</table>
  <div class="url-link">{url}</div>
  <div class="text-block" id="{text_id}">{html.escape(truncated)}</div>
  {toggle_html}
</details>
"""


sections_html = ""
jump_links    = ""
for qual in QUALITY_ORDER:
    anchor_id = f"qual_{str(qual).replace(' ', '_').replace('/', '_')}"
    jump_links += (f'<a href="#{anchor_id}">'
                   f'quality={html.escape(str(qual))}</a>\n')

    subset = df[df["content_quality"] == qual].reset_index(drop=True)
    cards  = "".join(render_card(row, f"{anchor_id}_{i}")
                     for i, (_, row) in enumerate(subset.iterrows()))

    sections_html += f"""
<div class="stratum-section" id="{anchor_id}">
  <div class="stratum-heading">content_quality = {html.escape(str(qual))}
      ({len(subset)} docs)</div>
  {cards}
</div>
"""

html_out = f"""<!DOCTYPE html>
<html lang="sv">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>propella-1 Swedish PDF inspection</title>
<style>{CSS}</style>
</head>
<body>
<h1>propella-1 × finepdfs — Swedish sample inspection</h1>
<div class="jump-links">{jump_links}</div>
{sections_html}
<script>{JS}</script>
</body>
</html>
"""

with open(INSPECT_HTML, "w", encoding="utf-8") as fh:
    fh.write(html_out)
print(f"  Wrote {os.path.getsize(INSPECT_HTML):,} bytes")


# ── 8. Cleanup info ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLEANUP INFO")
print("=" * 60)

hf_bytes = dir_size_bytes(HF_CACHE)
sp_bytes  = file_size_bytes(SAMPLE_PARQUET)
ih_bytes  = file_size_bytes(INSPECT_HTML)

print(f"\n  hf_cache/")
print(f"    path : {HF_CACHE}")
print(f"    size : {human_bytes(hf_bytes)}")
print(f"    delete: rm -rf {HF_CACHE}")

print(f"\n  sample.parquet")
print(f"    path : {SAMPLE_PARQUET}")
print(f"    size : {human_bytes(sp_bytes)}")
print(f"    delete: rm -rf {SAMPLE_PARQUET}")

print(f"\n  inspect.html")
print(f"    path : {INSPECT_HTML}")
print(f"    size : {human_bytes(ih_bytes)}")
print(f"    delete: rm -rf {INSPECT_HTML}")

print(f"\n  Delete everything at once:")
print(f"    rm -rf {HF_CACHE} {SAMPLE_PARQUET} {INSPECT_HTML}")
print()
