"""LLM-vs-Human diagnostic: score, compare, and visualize on the same 130 docs.

Step 1 — Score: LLM-score the 130 Human-annotated docs via gemma-4-31B on Berget.
         Skipped if outputs/human_subset_llm_scores.csv already exists.
         Requires BERGET_API_KEY in the environment.

Step 2 — Compare: join on canonical id, print correlations, top/bottom-k agreement,
         and per-doc disagreements.

Step 3 — Visualize: produce a 4-panel PNG (scatter, distributions, Bland-Altman,
         per-axis decomposition) → outputs/human_vs_llm_comparison.png.

quality_score is always the unweighted arithmetic mean of educational_value and
content_quality, recomputed from per-axis scores regardless of what the CSV stores.

Run from llm-scorer/:
    ../.venv/bin/python llm_human_diagnostic.py          # full pipeline
    ../.venv/bin/python llm_human_diagnostic.py --rescore # force re-scoring
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                      / "reranker" / "src"))
from learned_reranker.data import canonical_doc_id  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
HUMAN_CSV = HERE.parent / "reranker" / "traindata" / "Human_train.csv"
PROMPT_FILE = HERE / "prompts" / "quality_prompt.txt"
OUT_CSV = HERE / "outputs" / "human_subset_llm_scores.csv"
TEXT_CACHE = HERE / "outputs" / "human_subset_raw_texts.parquet"
PNG_OUT = HERE / "outputs" / "human_vs_llm_comparison.png"
TXT_OUT = HERE / "outputs" / "human_vs_llm_comparison.txt"
LANG_TAG = "swe_Latn_human_subset"


# ---------------------------------------------------------------------------
# Step 1: Score (or load cached scores)
# ---------------------------------------------------------------------------

def _fetch_texts(canon_ids: set[str]) -> dict[str, str]:
    """Fetch raw text for *canon_ids* via DuckDB+httpfs, with parquet cache."""
    from learned_reranker.config import DatasetConfig
    from learned_reranker.data import _finepdf_via_duckdb

    if TEXT_CACHE.exists():
        text_pd = pd.read_parquet(TEXT_CACHE)
        print(f"[score] Loaded text cache ({len(text_pd)} rows)")
    else:
        cfg = DatasetConfig()
        text_frame = _finepdf_via_duckdb(cfg, canon_ids, columns=("id", "text"))
        text_pd = text_frame.to_pandas()
        TEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        text_pd.to_parquet(TEXT_CACHE, index=False)
        print(f"[score] Fetched {len(text_pd)} docs → {TEXT_CACHE}")

    texts: dict[str, str] = {}
    for _, row in text_pd.iterrows():
        cid = canonical_doc_id(str(row["id"]))
        if row["text"]:
            texts[cid] = row["text"]
    print(f"[score] {len(texts)} non-empty, {len(canon_ids) - len(texts)} missing/empty")
    return texts


def _score_docs(canon_ids: set[str]) -> pd.DataFrame:
    """LLM-score documents and write to OUT_CSV."""
    from src.config import (
        DEFAULT_MAX_CHARS, DEFAULT_MAX_TOKENS, DEFAULT_PROVIDER,
        build_api_config, default_model_for,
    )
    from src.llm_scorer import score_documents

    texts = _fetch_texts(canon_ids)

    api_key_env = "BERGET_API_KEY"
    if not os.environ.get(api_key_env):
        sys.exit(f"[score] {api_key_env} not set — cannot score. "
                 f"Text cache is at {TEXT_CACHE}; set the key and re-run.")

    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    provider = DEFAULT_PROVIDER
    model = default_model_for(provider)
    api_cfg = build_api_config(provider, model, DEFAULT_MAX_TOKENS, 0.0)
    print(f"[score] Scoring {len(texts)} docs via {provider}/{model}, "
          f"max_chars={DEFAULT_MAX_CHARS}")
    scored = score_documents(
        texts, prompt, api_cfg,
        output_dir=str(OUT_CSV.parent),
        language=LANG_TAG,
        batch_size=5,
        max_chars=DEFAULT_MAX_CHARS,
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT_CSV, index=False)
    print(f"[score] Wrote {OUT_CSV} ({len(scored)} rows)")
    return scored


def load_or_score(h: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    """Return LLM scores DataFrame, scoring via API only if needed."""
    canon_ids = set(h["canon"])
    if OUT_CSV.exists() and not force:
        llm = pd.read_csv(OUT_CSV)
        print(f"[score] Using cached {OUT_CSV} ({len(llm)} rows)")
    else:
        llm = _score_docs(canon_ids)
    llm.drop(columns=["raw_response"], errors="ignore", inplace=True)
    llm["canon"] = llm["doc_id"].astype(str).map(canonical_doc_id)
    # Always recompute quality_score as unweighted mean.
    llm["quality_score"] = (llm["educational_value"] + llm["content_quality"]) / 2
    return llm


# ---------------------------------------------------------------------------
# Step 2: Compare (print text diagnostics)
# ---------------------------------------------------------------------------

def print_comparison(j: pd.DataFrame) -> None:
    """Print correlation, top/bottom-k agreement, and disagreements.

    Output goes to both stdout and TXT_OUT.
    """
    lines: list[str] = []

    def out(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    n = len(j)
    out(f"\n{'=' * 60}")
    out(f"  JOINED: {n} docs with both Human score AND LLM quality_score")
    out(f"{'=' * 60}\n")

    out(f"Human score    : mean={j.score.mean():.3f}  median={j.score.median():.3f}  "
        f"std={j.score.std():.3f}  min={j.score.min():.3f}  max={j.score.max():.3f}")
    out(f"LLM quality_sc : mean={j.quality_score.mean():.3f}  median={j.quality_score.median():.3f}  "
        f"std={j.quality_score.std():.3f}  min={j.quality_score.min():.3f}  max={j.quality_score.max():.3f}")
    out(f"  LLM edu axis : mean={j.llm_edu.mean():.3f}")
    out(f"  LLM qual axis: mean={j.llm_cq.mean():.3f}")

    pearson = stats.pearsonr(j.score, j.quality_score)
    spearman = stats.spearmanr(j.score, j.quality_score)
    kendall = stats.kendalltau(j.score, j.quality_score)
    out("\n=== correlations: Human score vs LLM quality_score ===")
    out(f"  Pearson  r = {pearson.statistic:+.3f}  (p={pearson.pvalue:.2e})")
    out(f"  Spearman ρ = {spearman.statistic:+.3f}  (p={spearman.pvalue:.2e})")
    out(f"  Kendall  τ = {kendall.statistic:+.3f}  (p={kendall.pvalue:.2e})")

    sp_edu = stats.spearmanr(j.score, j.llm_edu)
    sp_cq = stats.spearmanr(j.score, j.llm_cq)
    out(f"\n  vs LLM educational_value alone: ρ = {sp_edu.statistic:+.3f}")
    out(f"  vs LLM content_quality   alone: ρ = {sp_cq.statistic:+.3f}")

    k = max(1, int(n * 0.10))
    top_h = set(j.nlargest(k, "score")["canon"])
    top_l = set(j.nlargest(k, "quality_score")["canon"])
    bot_h = set(j.nsmallest(k, "score")["canon"])
    bot_l = set(j.nsmallest(k, "quality_score")["canon"])
    out(f"\n=== top/bottom-{k} (~10%) agreement ===")
    out(f"  top-{k}    overlap: {len(top_h & top_l)}/{k}  "
        f"(Jaccard {len(top_h & top_l) / len(top_h | top_l):.2f})")
    out(f"  bottom-{k} overlap: {len(bot_h & bot_l)}/{k}  "
        f"(Jaccard {len(bot_h & bot_l) / len(bot_h | bot_l):.2f})")

    j = j.copy()
    j["delta"] = j.score - j.quality_score
    out("\n=== 5 biggest disagreements where Human > LLM ===")
    for r in j.nlargest(5, "delta").itertuples():
        out(f"  Human={r.score:.3f}  LLM={r.quality_score:.3f}  "
            f"delta=+{r.delta:.3f}  id={r.canon[:20]}...")
    out("\n=== 5 biggest disagreements where LLM > Human ===")
    for r in j.nsmallest(5, "delta").itertuples():
        out(f"  Human={r.score:.3f}  LLM={r.quality_score:.3f}  "
            f"delta={r.delta:+.3f}  id={r.canon[:20]}...")

    TXT_OUT.parent.mkdir(parents=True, exist_ok=True)
    TXT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {TXT_OUT}")


# ---------------------------------------------------------------------------
# Step 3: Visualize (4-panel PNG)
# ---------------------------------------------------------------------------

def make_figure(j: pd.DataFrame) -> None:
    """Create 4-panel comparison figure → PNG_OUT."""
    n = len(j)
    # Columns already renamed: llm_edu, llm_cq; shorten quality_score for plotting.
    j = j.rename(columns={"quality_score": "llm_qs"})

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ---- Panel A: main scatter
    ax = axes[0, 0]
    x, y = j.score.values, j.llm_qs.values
    ax.scatter(x, y, alpha=0.55, s=35, color="C0", edgecolor="white", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="y = x (perfect agreement)")
    m, b = np.polyfit(x, y, 1)
    xs = np.array([0.0, 1.0])
    ax.plot(xs, m * xs + b, "-", color="crimson", linewidth=1.8,
            label=f"OLS fit: y = {m:.2f}x + {b:.2f}")
    r = stats.pearsonr(x, y).statistic
    rho = stats.spearmanr(x, y).statistic
    tau = stats.kendalltau(x, y).statistic
    ax.set_xlabel("Human pairwise score")
    ax.set_ylabel("LLM quality_score  (unweighted mean)")
    ax.set_title(f"A.  Per-doc agreement (n={n})\n"
                 f"Pearson r = {r:+.3f}    Spearman ρ = {rho:+.3f}    Kendall τ = {tau:+.3f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # ---- Panel B: marginal distributions
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 26)
    ax.hist(j.score, bins=bins, alpha=0.55,
            label=f"Human  (μ={j.score.mean():.2f}, σ={j.score.std():.2f})",
            color="C0", density=True, edgecolor="white", linewidth=0.5)
    ax.hist(j.llm_qs, bins=bins, alpha=0.55,
            label=f"LLM    (μ={j.llm_qs.mean():.2f}, σ={j.llm_qs.std():.2f})",
            color="C1", density=True, edgecolor="white", linewidth=0.5)
    ax.axvline(j.score.mean(), color="C0", linestyle=":", linewidth=1.5)
    ax.axvline(j.llm_qs.mean(), color="C1", linestyle=":", linewidth=1.5)
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_title("B.  Marginal distributions")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # ---- Panel C: Bland-Altman
    ax = axes[1, 0]
    mean_xy = (j.score + j.llm_qs) / 2
    delta = j.score - j.llm_qs
    ax.scatter(mean_xy, delta, alpha=0.55, s=35, color="C2", edgecolor="white", linewidth=0.5)
    bias = float(delta.mean())
    sd = float(delta.std())
    ax.axhline(bias, color="crimson", linewidth=1.5, label=f"bias = {bias:+.3f}")
    ax.axhline(bias + 1.96 * sd, color="crimson", linestyle="--", alpha=0.7,
               label=f"±1.96σ limits of agreement (σ={sd:.3f})")
    ax.axhline(bias - 1.96 * sd, color="crimson", linestyle="--", alpha=0.7)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("(Human + LLM) / 2")
    ax.set_ylabel("Human − LLM  (positive = Human scores higher)")
    ax.set_title("C.  Bland-Altman agreement plot")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # ---- Panel D: per-axis decomposition
    ax = axes[1, 1]
    rho_edu = stats.spearmanr(j.score, j.llm_edu).statistic
    rho_cq = stats.spearmanr(j.score, j.llm_cq).statistic
    rho_combined = stats.spearmanr(j.score, j.llm_qs).statistic
    ax.scatter(j.score, j.llm_edu, alpha=0.5, s=30, color="C3",
               edgecolor="white", linewidth=0.5,
               label=f"LLM educational_value (ρ={rho_edu:+.2f})")
    ax.scatter(j.score, j.llm_cq, alpha=0.5, s=30, color="C4",
               edgecolor="white", linewidth=0.5,
               label=f"LLM content_quality   (ρ={rho_cq:+.2f})")
    ax.set_xlabel("Human pairwise score")
    ax.set_ylabel("LLM axis value")
    ax.set_title(f"D.  Per-LLM-axis vs Human\n"
                 f"combined quality_score (unweighted): ρ={rho_combined:+.2f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Human pairwise vs LLM judge (gemma-4-31B-it) — {n} documents  "
        f"[quality_score = (edu + cq) / 2]",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PNG_OUT, dpi=120, bbox_inches="tight")
    print(f"\nWrote {PNG_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-vs-Human diagnostic")
    parser.add_argument("--rescore", action="store_true",
                        help="Force re-scoring even if cached CSV exists")
    args = parser.parse_args()

    h = pd.read_csv(HUMAN_CSV)
    h["canon"] = h["id"].astype(str).map(canonical_doc_id)

    llm = load_or_score(h, force=args.rescore)

    # Rename LLM axes to avoid collision with Human_train's ordinal columns.
    llm_subset = llm[["canon", "educational_value", "content_quality", "quality_score"]].rename(
        columns={"educational_value": "llm_edu", "content_quality": "llm_cq"}
    )
    j = h.merge(llm_subset, on="canon", how="inner").dropna(subset=["score", "quality_score"])

    if len(j) < 2:
        sys.exit(f"Only {len(j)} joined rows — not enough for analysis.")

    print_comparison(j)
    make_figure(j)


if __name__ == "__main__":
    main()
