"""
generate_scored_inspect.py
Fetches full_doc_lid, full_doc_lid_score, minhash_cluster_size, duplicate_count
from HuggingFaceFW/finepdfs for the 130 scored documents, then writes
scored_inspect.html — an inspect.html-style file without raw text.

Uses pyarrow column projection over the HuggingFace filesystem so only the
5 needed columns are fetched via HTTP range requests — no full-file downloads.
"""
import os, html, re
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / "hf_cache"))

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

SCORES_CSV    = SCRIPT_DIR / "document_scores.csv"
SAMPLE_PQ     = SCRIPT_DIR / "sample.parquet"
OUTPUT_HTML   = SCRIPT_DIR / "scored_inspect.html"

EXTRA_FIELDS  = ["full_doc_lid", "full_doc_lid_score", "minhash_cluster_size", "duplicate_count"]
FETCH_COLS    = ["id"] + EXTRA_FIELDS

# All 13 swe_Latn train shards
SHARD_PATHS = [
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/000_00000.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/000_00001.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/000_00002.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/000_00003.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/000_00004.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/000_00005.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/000_00006.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/001_00000.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/001_00001.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/001_00002.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/001_00003.parquet",
    "datasets/HuggingFaceFW/finepdfs/data/swe_Latn/train/001_00004.parquet",
]


def normalize_id(v: str) -> str:
    if v and v.startswith("<urn:uuid:") and v.endswith(">"):
        return v[10:-1]
    return str(v)


def coerce_str(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v) if v is not None else ""


def main():
    scores = pd.read_csv(SCORES_CSV)
    sample = pd.read_parquet(SAMPLE_PQ)

    scored_ids = set(normalize_id(str(i)) for i in scores["doc_id"])
    print(f"Scored documents: {len(scored_ids)}")

    # Normalize sample IDs for join
    sample["_norm_id"] = sample["id"].apply(lambda x: normalize_id(str(x)))

    # Use HuggingFace filesystem + pyarrow column projection.
    # Only the 5 requested columns are fetched via HTTP range requests —
    # no full-file downloads needed.
    print(f"\nFetching columns {FETCH_COLS} from {len(SHARD_PATHS)} shards via column projection…")
    fs = HfFileSystem()

    chunks = []
    for i, path in enumerate(SHARD_PATHS, 1):
        print(f"  Shard {i}/{len(SHARD_PATHS)}: {path.split('/')[-1]} … ", end="", flush=True)
        table = pq.read_table(path, filesystem=fs, columns=FETCH_COLS)
        df_shard = table.to_pandas()
        df_shard["_norm_id"] = df_shard["id"].apply(lambda x: normalize_id(str(x)))
        matched = df_shard[df_shard["_norm_id"].isin(scored_ids)]
        chunks.append(matched)
        print(f"{len(matched)} matched (shard rows: {len(df_shard):,})")

    extra_df = pd.concat(chunks, ignore_index=True).drop_duplicates("_norm_id")
    print(f"\nTotal matched: {len(extra_df)} / {len(scored_ids)}")
    missing = scored_ids - set(extra_df["_norm_id"])
    if missing:
        print(f"[WARNING] {len(missing)} IDs not found in any shard.")

    # Build merged dataframe: sample + scores + extra fields
    scores["_norm_id"] = scores["doc_id"].apply(lambda x: normalize_id(str(x)))
    df = sample.merge(scores[["_norm_id", "score", "wins", "losses", "n_comparisons"]],
                      on="_norm_id", how="inner")
    df = df.merge(extra_df, on="_norm_id", how="left")
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    print(f"\nMerged dataframe: {len(df)} rows")

    # ── Generate HTML ──────────────────────────────────────────────────────
    ANN_FIELDS = [
        "content_quality", "educational_value",
        "content_integrity", "content_ratio", "content_length",
        "content_type", "business_sector", "technical_content",
        "information_density", "audience_level", "commercial_bias",
        "time_sensitivity", "content_safety", "reasoning_indicators",
        "pii_presence", "regional_relevance", "country_relevance",
    ]
    SCORE_FIELDS = ["score", "wins", "losses", "n_comparisons"]
    EXTRA_DISPLAY = EXTRA_FIELDS  # full_doc_lid, full_doc_lid_score, minhash_cluster_size, duplicate_count

    CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; font-size: 14px;
       background: #f5f5f5; color: #222; padding: 1rem; }
h1 { margin-bottom: 0.4rem; font-size: 1.4rem; }
.subtitle { color: #666; font-size: 0.9rem; margin-bottom: 1.2rem; }
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
summary { cursor: pointer; font-weight: 600; font-size: 0.95rem; list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "▶ "; font-size: 0.7rem; color: #888; }
details[open] summary::before { content: "▼ "; }
.ann-table { width: 100%; border-collapse: collapse; margin: 0.7rem 0; font-size: 0.82rem; }
.ann-table th { text-align: left; padding: 3px 8px; background: #f0f0f0;
                border-bottom: 1px solid #ddd; white-space: nowrap; font-weight: 600; }
.ann-table td { padding: 3px 8px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
.section-scores { background: #eef4ff; }
.section-extra  { background: #f4f9f0; }
.ann-table tr.section-header th { background: #d0e4ff; font-size: 0.78rem;
                                   color: #003080; letter-spacing: 0.05em;
                                   text-transform: uppercase; padding-top: 6px; }
.ann-table tr.section-header-extra th { background: #d4edda; font-size: 0.78rem;
                                         color: #155724; letter-spacing: 0.05em;
                                         text-transform: uppercase; padding-top: 6px; }
.score-badge { display: inline-block; background: #0066cc; color: #fff;
               border-radius: 4px; padding: 0.15rem 0.5rem; font-size: 0.78rem;
               font-weight: 700; margin-left: 0.5rem; vertical-align: middle; }
.url-link { font-size: 0.75rem; color: #888; word-break: break-all; margin-top: 4px; }
.doc-id   { font-size: 0.72rem; color: #aaa; word-break: break-all; margin-top: 2px; }
"""

    def render_card(row):
        desc = html.escape(coerce_str(row.get("one_sentence_description", "")) or "(no description)")
        url  = html.escape(str(row.get("url", "") or ""))
        doc_id = html.escape(str(row.get("id", "") or ""))
        score_val = row.get("score", "")
        score_fmt = f"{score_val:.4f}" if isinstance(score_val, float) else str(score_val)

        rows_html = '<tr class="section-header"><th colspan="2">Propella-1 annotations</th></tr>\n'
        for field in ANN_FIELDS:
            val = coerce_str(row.get(field, ""))
            rows_html += (f"<tr><th>{html.escape(field)}</th>"
                          f"<td>{html.escape(val)}</td></tr>\n")

        rows_html += '<tr class="section-header"><th colspan="2">Human rating scores</th></tr>\n'
        for field in SCORE_FIELDS:
            val = coerce_str(row.get(field, ""))
            rows_html += (f"<tr><th>{html.escape(field)}</th>"
                          f"<td>{html.escape(val)}</td></tr>\n")

        rows_html += '<tr class="section-header-extra"><th colspan="2">finepdfs metadata</th></tr>\n'
        for field in EXTRA_DISPLAY:
            val = coerce_str(row.get(field, ""))
            rows_html += (f"<tr><th>{html.escape(field)}</th>"
                          f"<td>{html.escape(val)}</td></tr>\n")

        return f"""
<details>
  <summary>{desc}<span class="score-badge">{html.escape(score_fmt)}</span></summary>
  <table class="ann-table">{rows_html}</table>
  <div class="url-link">{url}</div>
  <div class="doc-id">id: {doc_id}</div>
</details>
"""

    # Group by content_quality, sorted best→worst within each group
    QUALITY_ORDER = ["excellent", "good", "adequate", "poor", "unacceptable"]
    present_qualities = [q for q in QUALITY_ORDER if q in df["content_quality"].values]
    # Also catch any unlisted values
    for q in df["content_quality"].unique():
        if q not in present_qualities:
            present_qualities.append(q)

    jump_links = ""
    sections_html = ""
    for qual in present_qualities:
        anchor_id = f"qual_{re.sub(r'[^a-z0-9]', '_', str(qual).lower())}"
        subset = df[df["content_quality"] == qual].sort_values("score", ascending=False).reset_index(drop=True)
        jump_links += f'<a href="#{anchor_id}">{html.escape(str(qual))} ({len(subset)})</a>\n'
        cards = "".join(render_card(row) for _, row in subset.iterrows())
        sections_html += f"""
<div class="stratum-section" id="{anchor_id}">
  <div class="stratum-heading">content_quality = {html.escape(str(qual))} — {len(subset)} documents (sorted by score ↓)</div>
  {cards}
</div>
"""

    html_out = f"""<!DOCTYPE html>
<html lang="sv">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scored document inspection — 130 rated documents</title>
<style>{CSS}</style>
</head>
<body>
<h1>Scored document inspection</h1>
<p class="subtitle">130 documents with human A/B test scores · sorted by score (high → low) within each quality group · no raw text</p>
<div class="jump-links">{jump_links}</div>
{sections_html}
</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html_out)
    print(f"\nWrote {OUTPUT_HTML.stat().st_size:,} bytes to {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
