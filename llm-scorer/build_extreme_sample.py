"""Build extreme-quality samples, score with LLM judge, and produce training sets.

1. Stream Propella swe_Latn, filter into top/bottom pools:
     Top    : edu>=3 AND cq>=3 AND content_ratio==complete_content
              AND content_safety==safe AND pii_presence==no_pii     → sample 1000 (seed 42)
     Bottom : edu<=1 AND cq<=2 AND content_ratio in
              {mostly_navigation, minimal_content, mixed_content}   → sample 1000 (seed 42)
2. Fetch text from FinePDFs.
3. Score with LLM judge.
4. Filter scored docs by strict Propella ordinals into training sets:
     Top    : edu==4 AND cq==4
     Bottom : edu<=1 AND cq<=1

Exclusions (BOTH pools):
  outputs/LLM_scoring_finepdf_propella_combined_{lang}.csv  (already-scored LLM docs)
  ../reranker/traindata/Human_train.csv                     (held-out 130-doc human eval set)

Outputs:
  outputs/extreme_sample_{top,bottom}.csv   — canonical 24 cols
  outputs/extreme_sample_raw_texts.parquet  — id, text, pool
  outputs/extreme_sample_scored.csv         — scored canonical + LLM per-axis scores
  outputs/extreme_train_top.csv             — final training set (top)
  outputs/extreme_train_bottom.csv          — final training set (bottom)

Run from llm-gold-standard/:
    ../.venv/bin/python build_extreme_sample.py                  # build + score + train sets
    ../.venv/bin/python build_extreme_sample.py --skip_scoring   # build only
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import pandas as pd
from datasets import load_dataset
from scipy import stats

from src.finepdfs import _normalize_id, _scan_finepdfs
from src.config import (
    CANONICAL_COLUMNS,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROVIDER,
    FINEPDFS_META,
    ORDINAL_MAPS,
    PROPELLA_COLS,
    PROPELLA_DATASET,
    PROPELLA_SUBSET,
    build_api_config,
    default_model_for,
)
from src.llm_scorer import score_documents
from src.utils import log

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                      / "reranker" / "src"))
from learned_reranker.data import canonical_doc_id  # noqa: E402

LANGUAGE = DEFAULT_LANGUAGE  # swe_Latn
OUTPUT_DIR = pathlib.Path(DEFAULT_OUTPUT_DIR)
EXCLUDE_LLM = OUTPUT_DIR / f"LLM_scoring_finepdf_propella_combined_{LANGUAGE}.csv"
EXCLUDE_HUMAN = pathlib.Path("../reranker/traindata/Human_train.csv")
TOP_OUT = OUTPUT_DIR / "extreme_sample_top.csv"
BOTTOM_OUT = OUTPUT_DIR / "extreme_sample_bottom.csv"
TEXT_CACHE = OUTPUT_DIR / "extreme_sample_raw_texts.parquet"
PROMPT_FILE = pathlib.Path("prompts/quality_prompt.txt")
SCORED_OUT = OUTPUT_DIR / "extreme_sample_scored.csv"
TRAIN_TOP_OUT = OUTPUT_DIR / "extreme_train_top.csv"
TRAIN_BOT_OUT = OUTPUT_DIR / "extreme_train_bottom.csv"
LANG_TAG = "swe_Latn_extreme"  # namespaces the partial parquet so it never collides

EDU = ORDINAL_MAPS["educational_value"]   # none=0 ... high=4
CQ = ORDINAL_MAPS["content_quality"]      # unacceptable=0 ... excellent=4

TOP_RATIO = "complete_content"
TOP_SAFETY = "safe"
TOP_PII = "no_pii"
BOTTOM_RATIO = {"mostly_navigation", "minimal_content", "mixed_content"}

TARGET_PER_POOL = 1000
SEED = 42

# Canonical CSV schema: id, quality_score placeholder, 18 Propella features, 4 FinePDFs metadata.
CANONICAL = CANONICAL_COLUMNS
# Appended training cols.
TRAIN_APPENDED = ["gold_score", "llm_educational_value", "llm_content_quality", "source"]
TRAIN_COLS = CANONICAL + TRAIN_APPENDED


def _ordinal(label, mapping):
    if not label:
        return None
    return mapping.get(str(label).strip().lower())


def _to_ord(value, mapping: dict) -> int | None:
    """Coerce a Propella ordinal to int. Handles strings ('high'), ints (4), and NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        try:
            iv = int(value)
            return iv if 0 <= iv <= 4 else None
        except (TypeError, ValueError):
            return None
    return mapping.get(str(value).strip().lower())


def _score_and_report(combined: pd.DataFrame, text_cache: pathlib.Path,
                      api_cfg: dict, batch_size: int, max_chars: int) -> None:
    """Score the built samples with the LLM judge and write the scored CSV."""
    text_df = pd.read_parquet(text_cache)
    text_by_id = dict(zip(text_df["id"].astype(str), text_df["text"].astype(str)))
    combined["text"] = combined["id"].astype(str).map(text_by_id).fillna("")
    log(f"Text cache: {len(text_df)} rows -> "
        f"{(combined.text.str.len() > 0).sum()} rows with text")

    nonempty = combined[combined["text"].str.len() > 0].copy()
    dropped = len(combined) - len(nonempty)
    log(f"Non-empty text: {len(nonempty)} "
        f"(dropped {dropped} empty-text rows: "
        f"{(combined.pool == 'top').sum() - (nonempty.pool == 'top').sum()} top, "
        f"{(combined.pool == 'bottom').sum() - (nonempty.pool == 'bottom').sum()} bottom)")

    texts = dict(zip(nonempty["id"].astype(str), nonempty["text"].astype(str)))
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    log(f"Scoring {len(texts)} docs via {api_cfg['provider']}/{api_cfg['model']} "
        f"(max_chars={max_chars}, batch_size={batch_size})...")
    scored = score_documents(
        texts, prompt, api_cfg,
        output_dir=str(SCORED_OUT.parent),
        language=LANG_TAG,
        batch_size=batch_size,
        max_chars=max_chars,
    )
    log(f"Scoring complete: {len(scored)} rows returned")

    # Join LLM scores onto the canonical metadata. Rename to avoid Propella/LLM
    # column collisions: Propella `educational_value`/`content_quality` are ordinal strings;
    # the LLM ones are 0-1 floats.
    scored = scored.rename(columns={
        "doc_id": "id",
        "educational_value": "llm_educational_value",
        "content_quality": "llm_content_quality",
        "quality_score": "llm_quality_score",
    })
    keep_llm = ["id", "llm_educational_value", "llm_content_quality",
                "llm_quality_score", "raw_response"]
    canonical_no_text = nonempty.drop(columns=["text"])
    out = canonical_no_text.merge(scored[keep_llm], on="id", how="left")
    out["quality_score"] = out["llm_quality_score"]
    out.to_csv(SCORED_OUT, index=False)
    log(f"Wrote {SCORED_OUT} ({len(out)} rows)")

    # Top-vs-bottom comparison report.
    print("\n" + "=" * 72)
    print("LLM judge on the extreme-quality sample")
    print("=" * 72)
    for label in ("top", "bottom"):
        sub = out[(out["pool"] == label) & out["llm_quality_score"].notna()]
        if not len(sub):
            print(f"\nPool '{label}': no scored rows")
            continue
        print(f"\nPool '{label}' (n={len(sub)}):")
        print(f"  llm_quality_score  mean={sub.llm_quality_score.mean():.3f}  "
              f"median={sub.llm_quality_score.median():.3f}  "
              f"std={sub.llm_quality_score.std():.3f}  "
              f"min={sub.llm_quality_score.min():.3f}  max={sub.llm_quality_score.max():.3f}")
        print(f"  llm_educational    mean={sub.llm_educational_value.mean():.3f}  "
              f"std={sub.llm_educational_value.std():.3f}")
        print(f"  llm_content_quality mean={sub.llm_content_quality.mean():.3f}  "
              f"std={sub.llm_content_quality.std():.3f}")

    t = out[(out.pool == "top") & out.llm_quality_score.notna()]["llm_quality_score"]
    b = out[(out.pool == "bottom") & out.llm_quality_score.notna()]["llm_quality_score"]
    if len(t) and len(b):
        gap = float(t.mean() - b.mean())
        u, p = stats.mannwhitneyu(t, b, alternative="greater")
        auc = float(u / (len(t) * len(b)))
        print(f"\nGap (top mean - bottom mean): {gap:+.3f}")
        print(f"AUC = P(random_top > random_bottom on llm_quality_score): {auc:.3f}  "
              f"(1.0 = perfect separation, 0.5 = random)")
        print(f"Mann-Whitney U p-value (one-sided top>bottom): {p:.2e}")


def _build_train_sets() -> None:
    """Filter scored extreme samples into final training sets by strict Propella ordinals."""
    src = pd.read_csv(SCORED_OUT)
    src["source"] = "extreme_sample_" + src["pool"].astype(str)
    src = src.drop(columns=["pool", "raw_response", "llm_quality_score"], errors="ignore")
    log(f"Building train sets from {len(src)} scored docs")

    # Dedupe by canonical id.
    src["_canon"] = src["id"].astype(str).map(canonical_doc_id)
    before = len(src)
    combined = src.drop_duplicates(subset=["_canon"], keep="first").reset_index(drop=True)
    if before != len(combined):
        log(f"Dedupe: {before} -> {len(combined)} (removed {before - len(combined)})")

    # Exclude held-out human-eval ids.
    if EXCLUDE_HUMAN.exists():
        human_canon = {canonical_doc_id(str(x))
                       for x in pd.read_csv(EXCLUDE_HUMAN, usecols=["id"])["id"]}
        before = len(combined)
        combined = combined[~combined["_canon"].isin(human_canon)].reset_index(drop=True)
        if before != len(combined):
            log(f"Excluded {before - len(combined)} Human_train ids")

    # Coerce Propella ordinals to int for filtering.
    combined["edu_ord"] = combined["educational_value"].map(lambda v: _to_ord(v, EDU))
    combined["cq_ord"] = combined["content_quality"].map(lambda v: _to_ord(v, CQ))
    bad = combined[combined["edu_ord"].isna() | combined["cq_ord"].isna()]
    if len(bad):
        log(f"WARNING: {len(bad)} docs with unparseable ordinals — excluded.")
        combined = combined.dropna(subset=["edu_ord", "cq_ord"]).reset_index(drop=True)

    # Filter into top and bottom.
    top = combined[(combined["edu_ord"] == 4) & (combined["cq_ord"] == 4)].copy()
    bot = combined[(combined["edu_ord"] <= 1) & (combined["cq_ord"] <= 1)].copy()

    # gold_score := llm_content_quality.
    for df in (top, bot):
        df["gold_score"] = df["llm_content_quality"]

    # Project to canonical 24 + appended training cols.
    def _project(df: pd.DataFrame) -> pd.DataFrame:
        for c in TRAIN_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df[TRAIN_COLS]

    top_out = _project(top)
    bot_out = _project(bot)
    top_out.to_csv(TRAIN_TOP_OUT, index=False)
    bot_out.to_csv(TRAIN_BOT_OUT, index=False)

    print("\n" + "=" * 72)
    print("TRAIN SETS")
    print("=" * 72)
    print(f"  {TRAIN_TOP_OUT.name}:    {len(top_out)} rows  (edu==4 AND cq==4)")
    print(f"  {TRAIN_BOT_OUT.name}: {len(bot_out)} rows  (edu<=1 AND cq<=1)")

    for label, df in [("TOP", top_out), ("BOTTOM", bot_out)]:
        if not len(df):
            continue
        ll_cq = df["llm_content_quality"].dropna()
        ll_edu = df["llm_educational_value"].dropna()
        ll_qs = df["quality_score"].dropna()
        print(f"\n{label} ({len(df)} docs):")
        print(f"  llm_content_quality    mean={ll_cq.mean():.3f}  std={ll_cq.std():.3f}")
        print(f"  llm_educational_value  mean={ll_edu.mean():.3f}  std={ll_edu.std():.3f}")
        print(f"  quality_score (combined) mean={ll_qs.mean():.3f}  std={ll_qs.std():.3f}")
        print(f"  source: {dict(df['source'].value_counts())}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build + score extreme-quality samples + training sets.")
    p.add_argument("--skip_scoring", action="store_true",
                   help="build only — skip the LLM scoring step")
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--anthropic", action="store_true",
                   help="shortcut for --provider anthropic (uses Claude)")
    p.add_argument("--model", default=None)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--max_chars", type=int, default=DEFAULT_MAX_CHARS)
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------- Step 1: stream Propella swe_Latn, filter into top/bottom pools.
    log(f"Streaming {PROPELLA_DATASET}/{PROPELLA_SUBSET} split='{LANGUAGE}' (no cap)...")
    ds = load_dataset(PROPELLA_DATASET, PROPELLA_SUBSET, split=LANGUAGE, streaming=True)

    top_pool: list[dict] = []
    bottom_pool: list[dict] = []
    n = 0
    for row in ds:
        n += 1
        edu_n = _ordinal(row.get("educational_value"), EDU)
        cq_n = _ordinal(row.get("content_quality"), CQ)
        if edu_n is None or cq_n is None:
            if n % 25_000 == 0:
                log(f"  Propella scanned {n:,}: top={len(top_pool):,} bottom={len(bottom_pool):,}")
            continue
        safety = row.get("content_safety")
        pii = row.get("pii_presence")
        ratio = row.get("content_ratio")
        propella = {"id": row.get("id"), **{c: row.get(c) for c in PROPELLA_COLS}}
        if (edu_n >= 3 and cq_n >= 3 and ratio == TOP_RATIO
                and safety == TOP_SAFETY and pii == TOP_PII):
            top_pool.append(propella)
        elif edu_n <= 1 and cq_n <= 2 and ratio in BOTTOM_RATIO:
            bottom_pool.append(propella)
        if n % 25_000 == 0:
            log(f"  Propella scanned {n:,}: top={len(top_pool):,} bottom={len(bottom_pool):,}")
    log(f"Propella stream complete: {n:,} rows scanned.")
    log(f"Pools (pre-exclusion): top={len(top_pool):,}  bottom={len(bottom_pool):,}")

    # -------- Step 2: build exclusion set (LLM-scored + Human held-out).
    excl_norm: set[str] = set()
    for src_path, label in [(EXCLUDE_LLM, "LLM-scored"),
                            (EXCLUDE_HUMAN, "Human eval set")]:
        if src_path.exists():
            ids = pd.read_csv(src_path, usecols=["id"])["id"].astype(str)
            added = {_normalize_id(x) for x in ids}
            new = len(added - excl_norm)
            excl_norm.update(added)
            log(f"  Exclusion source [{label}]: {len(ids):,} ids ({new:,} new) from {src_path}")
        else:
            log(f"  WARNING: exclusion source not found at {src_path} — skipping ({label})")
    log(f"Total exclusion set: {len(excl_norm):,} unique normalized ids")

    def _drop_excluded(pool):
        return [r for r in pool if _normalize_id(r["id"]) not in excl_norm]

    top_kept = _drop_excluded(top_pool)
    bot_kept = _drop_excluded(bottom_pool)
    log(f"After exclusion: top={len(top_kept):,}  bottom={len(bot_kept):,}")

    # -------- Step 3: sample 1000 from each pool with seed 42.
    def _sample(df, label):
        if len(df) > TARGET_PER_POOL:
            return df.sample(n=TARGET_PER_POOL, random_state=SEED).reset_index(drop=True)
        log(f"  WARNING: {label} pool ({len(df):,}) below target {TARGET_PER_POOL}; taking all.")
        return df.reset_index(drop=True)

    top_sample = _sample(pd.DataFrame(top_kept), "top")
    bot_sample = _sample(pd.DataFrame(bot_kept), "bottom")
    log(f"Sampled: top={len(top_sample):,}  bottom={len(bot_sample):,}")

    # -------- Step 4: FinePDFs fetch (text + 4 metadata cols) via the streaming scan.
    target_ids = pd.concat([top_sample["id"], bot_sample["id"]])
    target_norm = {_normalize_id(i) for i in target_ids}
    log(f"FinePDFs streaming scan for {len(target_norm):,} unique ids "
        f"(early-termination when all matched; this can take a long time)...")
    found, scanned = _scan_finepdfs(target_norm, LANGUAGE, columns=["text", *FINEPDFS_META])
    log(f"FinePDFs scan done: matched {len(found):,}/{len(target_norm):,} after {scanned:,} rows.")

    text_by_norm = {nid: (d.get("text") or "") for nid, d in found.items()}

    def build_canonical(df):
        """Canonical CSV (no text): id, quality_score placeholder, 18 Propella, 4 FinePDFs meta."""
        df = df.copy()
        df["_norm"] = df["id"].astype(str).map(_normalize_id)
        df["quality_score"] = ""  # placeholder; filled by the LLM scoring pass.
        for c in FINEPDFS_META:
            df[c] = df["_norm"].map({nid: d.get(c) for nid, d in found.items()})
        return df[CANONICAL]

    def text_rows(df, pool):
        """id, text, pool — the text cache the LLM scoring pass reads."""
        df = df.copy()
        df["_norm"] = df["id"].astype(str).map(_normalize_id)
        return pd.DataFrame({"id": df["id"],
                             "text": df["_norm"].map(text_by_norm).fillna(""),
                             "pool": pool})

    top_out = build_canonical(top_sample)
    bot_out = build_canonical(bot_sample)
    text_df = pd.concat([text_rows(top_sample, "top"), text_rows(bot_sample, "bottom")],
                        ignore_index=True)

    # -------- Step 5: write outputs (canonical CSVs + text parquet) + summary.
    top_out.to_csv(TOP_OUT, index=False)
    bot_out.to_csv(BOTTOM_OUT, index=False)
    text_df.to_parquet(TEXT_CACHE, index=False)
    log(f"Wrote {TOP_OUT.name} ({len(top_out):,} rows), {BOTTOM_OUT.name} ({len(bot_out):,} rows), "
        f"{TEXT_CACHE.name} ({len(text_df):,} rows)")

    print("\n" + "=" * 72)
    print("BUILD SUMMARY")
    print("=" * 72)
    print(f"Propella rows scanned         : {n:,}")
    print(f"Top pool    pre-exclusion     : {len(top_pool):,}")
    print(f"Bottom pool pre-exclusion     : {len(bottom_pool):,}")
    print(f"Exclusion set (LLM + Human)   : {len(excl_norm):,}")
    print(f"Top pool    after exclusion   : {len(top_kept):,}")
    print(f"Bottom pool after exclusion   : {len(bot_kept):,}")
    print(f"Top sample  (target 1000)     : {len(top_sample):,}")
    print(f"Bottom sample (took all)      : {len(bot_sample):,}")
    print(f"FinePDFs matched              : {len(found):,}/{len(target_norm):,}  "
          f"(scanned {scanned:,})")
    nonempty_top = int((text_df[text_df.pool == "top"].text.str.len() > 0).sum())
    nonempty_bot = int((text_df[text_df.pool == "bottom"].text.str.len() > 0).sum())
    print(f"Top final with text           : {nonempty_top:,}/{len(top_out):,}")
    print(f"Bottom final with text        : {nonempty_bot:,}/{len(bot_out):,}")

    for label, df in [("TOP", top_out), ("BOTTOM", bot_out)]:
        print(f"\n  {label} sample distributions:")
        for c in ["educational_value", "content_quality", "content_ratio",
                  "content_safety", "pii_presence"]:
            d = dict(df[c].value_counts(dropna=False))
            print(f"    {c:20s} {d}")

    # -------- Step 6: LLM scoring (unless --skip_scoring).
    if args.skip_scoring:
        log("--skip_scoring: done (build only).")
        return

    provider = "anthropic" if args.anthropic else args.provider
    model = args.model or default_model_for(provider)
    key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else None
    api_cfg = build_api_config(provider, model, DEFAULT_MAX_TOKENS, 0.0,
                               api_key_env=key_env)

    # Skip cleanly if no API key (so the build work survives).
    key_var = api_cfg["api_key_env"]
    if not os.environ.get(key_var):
        log(f"{key_var} not set — stopping before LLM scoring. "
            f"Re-run with the key set to score.")
        return

    combined = pd.concat([
        top_out.assign(pool="top"),
        bot_out.assign(pool="bottom"),
    ], ignore_index=True)
    _score_and_report(combined, TEXT_CACHE, api_cfg, args.batch_size, args.max_chars)

    # -------- Step 7: build final training sets from scored data.
    _build_train_sets()


if __name__ == "__main__":
    main()
