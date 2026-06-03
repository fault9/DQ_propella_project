"""Compare LLM judge scores against Human pairwise scores on the SAME documents.

Human_train.csv and LLM_train.csv score essentially disjoint document sets (1/130
overlap), so a direct join answers nothing. This script LLM-scores the 130
Human-annotated docs, then joins on canonical id and prints correlation /
top-k-agreement / per-doc disagreement against the human pairwise `score`.

Run from llm-gold-standard/:
    ../.venv/bin/python compare_human_llm_scores.py
"""
from __future__ import annotations

import os
import pathlib
import sys

import pandas as pd
from scipy import stats

# Reuse the reranker's DuckDB FinePDFs fetcher + canonical_doc_id.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                      / "reranker" / "src"))
from learned_reranker.config import DatasetConfig  # noqa: E402
from learned_reranker.data import canonical_doc_id, _finepdf_via_duckdb  # noqa: E402

from src.config import (  # noqa: E402
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROVIDER,
    build_api_config,
    default_model_for,
)
from src.llm_scorer import score_documents  # noqa: E402


HUMAN_CSV = pathlib.Path(__file__).resolve().parent.parent / "reranker" / "traindata" / "Human_train.csv"
PROMPT_FILE = pathlib.Path(__file__).resolve().parent / "prompts" / "quality_prompt.txt"
OUT_CSV = pathlib.Path(__file__).resolve().parent / "outputs" / "human_subset_llm_scores.csv"
TEXT_CACHE = pathlib.Path(__file__).resolve().parent / "outputs" / "human_subset_raw_texts.parquet"
LANG_TAG = "swe_Latn_human_subset"


def main() -> None:
    h = pd.read_csv(HUMAN_CSV)
    h["canon"] = h["id"].astype(str).map(canonical_doc_id)
    canon_ids = set(h["canon"])
    print(f"[compare] Human ids: {len(canon_ids)} unique ({len(h)} rows)", flush=True)

    # ---- 1. Fetch raw text for the 130 ids via DuckDB+httpfs (or load from cache).
    if TEXT_CACHE.exists():
        text_pd = pd.read_parquet(TEXT_CACHE)
        print(f"[compare] Loaded text cache from {TEXT_CACHE} ({len(text_pd)} rows)", flush=True)
    else:
        cfg = DatasetConfig()  # defaults match what we need (swe_Latn / train).
        text_frame = _finepdf_via_duckdb(cfg, canon_ids, columns=("id", "text"))
        text_pd = text_frame.to_pandas()
        TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        text_pd.to_parquet(TEXT_CACHE, index=False)
        print(f"[compare] Fetched {len(text_pd)} docs and cached to {TEXT_CACHE}", flush=True)

    texts: dict[str, str] = {}
    for _, row in text_pd.iterrows():
        cid = canonical_doc_id(str(row["id"]))
        if row["text"]:
            texts[cid] = row["text"]
    missing = canon_ids - set(texts)
    print(f"[compare] Built texts dict: {len(texts)} non-empty, {len(missing)} missing/empty", flush=True)

    # Skip the LLM step cleanly if the API key isn't present (so the cached text
    # survives and a subsequent run with the key set picks up where we left off).
    api_key_env = "BERGET_API_KEY"  # matches provider preset; checked early.
    if not os.environ.get(api_key_env):
        print(f"\n[compare] {api_key_env} not set in environment — stopping before LLM scoring.")
        print(f"[compare] Text cache is at {TEXT_CACHE}; re-run with the key set to score "
              f"and compare.")
        return

    # ---- 2. LLM-score with the project default judge (gemma-4-31B on Berget).
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    provider = DEFAULT_PROVIDER
    model = default_model_for(provider)
    api_cfg = build_api_config(provider, model, DEFAULT_MAX_TOKENS, 0.0)
    print(f"[compare] Scoring {len(texts)} docs via {provider}/{model}, "
          f"max_chars={DEFAULT_MAX_CHARS}", flush=True)
    scored = score_documents(
        texts, prompt, api_cfg,
        output_dir=str(OUT_CSV.parent),
        language=LANG_TAG,
        batch_size=5,
        max_chars=DEFAULT_MAX_CHARS,
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT_CSV, index=False)
    print(f"[compare] Wrote {OUT_CSV} with {len(scored)} rows", flush=True)

    # ---- 3. Join on canonical id and compare.
    scored["canon"] = scored["doc_id"].astype(str).map(canonical_doc_id)
    j = h.merge(
        scored[["canon", "educational_value", "content_quality", "quality_score"]],
        on="canon", how="inner",
    ).dropna(subset=["score", "quality_score"])
    print(f"\n=== JOINED: {len(j)} docs with both Human score AND LLM quality_score ===\n", flush=True)
    if len(j) < 2:
        print("Not enough joined rows for correlation analysis.")
        return

    print(f"Human score    : mean={j.score.mean():.3f}  median={j.score.median():.3f}  "
          f"std={j.score.std():.3f}  min={j.score.min():.3f}  max={j.score.max():.3f}")
    print(f"LLM quality_sc : mean={j.quality_score.mean():.3f}  median={j.quality_score.median():.3f}  "
          f"std={j.quality_score.std():.3f}  min={j.quality_score.min():.3f}  max={j.quality_score.max():.3f}")
    print(f"  LLM edu axis : mean={j.educational_value.mean():.3f}")
    print(f"  LLM qual axis: mean={j.content_quality.mean():.3f}")

    pearson = stats.pearsonr(j.score, j.quality_score)
    spearman = stats.spearmanr(j.score, j.quality_score)
    kendall = stats.kendalltau(j.score, j.quality_score)
    print("\n=== correlations: Human score vs LLM quality_score ===")
    print(f"  Pearson  r = {pearson.statistic:+.3f}  (p={pearson.pvalue:.2e})")
    print(f"  Spearman ρ = {spearman.statistic:+.3f}  (p={spearman.pvalue:.2e})")
    print(f"  Kendall  τ = {kendall.statistic:+.3f}  (p={kendall.pvalue:.2e})")

    sp_edu = stats.spearmanr(j.score, j.educational_value)
    sp_cq = stats.spearmanr(j.score, j.content_quality)
    print(f"\n  vs LLM educational_value alone: ρ = {sp_edu.statistic:+.3f}")
    print(f"  vs LLM content_quality   alone: ρ = {sp_cq.statistic:+.3f}")

    k = max(1, int(len(j) * 0.10))
    top_h = set(j.nlargest(k, "score")["canon"])
    top_l = set(j.nlargest(k, "quality_score")["canon"])
    bot_h = set(j.nsmallest(k, "score")["canon"])
    bot_l = set(j.nsmallest(k, "quality_score")["canon"])
    print(f"\n=== top/bottom-{k} (~10%) agreement ===")
    print(f"  top-{k}    overlap: {len(top_h & top_l)}/{k}  (Jaccard {len(top_h & top_l)/len(top_h | top_l):.2f})")
    print(f"  bottom-{k} overlap: {len(bot_h & bot_l)}/{k}  (Jaccard {len(bot_h & bot_l)/len(bot_h | bot_l):.2f})")

    j["delta"] = j.score - j.quality_score
    print("\n=== 5 biggest disagreements where Human > LLM ===")
    for r in j.nlargest(5, "delta").itertuples():
        print(f"  Human={r.score:.3f}  LLM={r.quality_score:.3f}  delta=+{r.delta:.3f}  id={r.canon[:20]}...")
    print("\n=== 5 biggest disagreements where LLM > Human ===")
    for r in j.nsmallest(5, "delta").itertuples():
        print(f"  Human={r.score:.3f}  LLM={r.quality_score:.3f}  delta={r.delta:+.3f}  id={r.canon[:20]}...")


if __name__ == "__main__":
    main()
