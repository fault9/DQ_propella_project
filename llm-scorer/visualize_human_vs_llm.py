"""Four-panel visualization of Human pairwise vs LLM (gemma) scores on the same 130 docs.

Reads Human_train.csv + outputs/human_subset_llm_scores.csv, joins on canonical id,
writes outputs/human_vs_llm_comparison.png.

Panels:
  A. Per-doc scatter: Human score vs LLM quality_score, with y=x and best-fit line.
  B. Marginal distributions: histogram overlay.
  C. Bland-Altman agreement: mean vs delta, with bias and 95% limits of agreement.
  D. Per-LLM-axis decomposition: Human vs LLM educational_value and content_quality.

Run from llm-gold-standard/:
    ../.venv/bin/python visualize_human_vs_llm.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                      / "reranker" / "src"))
from learned_reranker.data import canonical_doc_id  # noqa: E402

HUMAN_CSV = pathlib.Path(__file__).resolve().parent.parent / "reranker" / "traindata" / "Human_train.csv"
LLM_CSV = pathlib.Path(__file__).resolve().parent / "outputs" / "human_subset_llm_scores.csv"
PNG_OUT = pathlib.Path(__file__).resolve().parent / "outputs" / "human_vs_llm_comparison.png"


def main() -> None:
    h = pd.read_csv(HUMAN_CSV)
    l = pd.read_csv(LLM_CSV)
    h["canon"] = h["id"].astype(str).map(canonical_doc_id)
    l["canon"] = l["doc_id"].astype(str).map(canonical_doc_id)
    # Rename LLM axis columns so they don't collide with Human_train's Propella ordinals
    # of the same name (the LLM ones are 0-1 floats; Human's are 0-4 ordinals).
    llm_subset = l[["canon", "educational_value", "content_quality", "quality_score"]].rename(
        columns={"educational_value": "llm_edu", "content_quality": "llm_cq",
                 "quality_score": "llm_quality_score"}
    )
    j = h.merge(llm_subset, on="canon", how="inner").dropna(subset=["score", "llm_quality_score"])
    n = len(j)
    print(f"Joined {n} docs")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ---- Panel A: main scatter (Human vs LLM quality_score)
    ax = axes[0, 0]
    x, y = j.score.values, j.llm_quality_score.values
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
    ax.set_ylabel("LLM (gemma) quality_score")
    ax.set_title(f"A.  Per-doc agreement (n={n})\n"
                 f"Pearson r = {r:+.3f}    Spearman ρ = {rho:+.3f}    Kendall τ = {tau:+.3f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # ---- Panel B: marginal distributions
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 26)
    ax.hist(j.score, bins=bins, alpha=0.55, label=f"Human  (μ={j.score.mean():.2f}, σ={j.score.std():.2f})",
            color="C0", density=True, edgecolor="white", linewidth=0.5)
    ax.hist(j.llm_quality_score, bins=bins, alpha=0.55,
            label=f"LLM    (μ={j.llm_quality_score.mean():.2f}, σ={j.llm_quality_score.std():.2f})",
            color="C1", density=True, edgecolor="white", linewidth=0.5)
    ax.axvline(j.score.mean(), color="C0", linestyle=":", linewidth=1.5)
    ax.axvline(j.llm_quality_score.mean(), color="C1", linestyle=":", linewidth=1.5)
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_title("B.  Marginal distributions")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # ---- Panel C: Bland-Altman (agreement)
    ax = axes[1, 0]
    mean_xy = (j.score + j.llm_quality_score) / 2
    delta = j.score - j.llm_quality_score
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

    # ---- Panel D: per-axis decomposition (which LLM axis tracks Human better?)
    ax = axes[1, 1]
    rho_edu = stats.spearmanr(j.score, j.llm_edu).statistic
    rho_cq = stats.spearmanr(j.score, j.llm_cq).statistic
    rho_combined = stats.spearmanr(j.score, j.llm_quality_score).statistic
    ax.scatter(j.score, j.llm_edu, alpha=0.5, s=30, color="C3",
               edgecolor="white", linewidth=0.5,
               label=f"LLM educational_value (ρ={rho_edu:+.2f})")
    ax.scatter(j.score, j.llm_cq, alpha=0.5, s=30, color="C4",
               edgecolor="white", linewidth=0.5,
               label=f"LLM content_quality   (ρ={rho_cq:+.2f})")
    ax.set_xlabel("Human pairwise score")
    ax.set_ylabel("LLM axis value")
    ax.set_title(f"D.  Per-LLM-axis vs Human\n"
                 f"combined quality_score: ρ={rho_combined:+.2f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Human pairwise vs LLM judge (gemma-4-31B-it) on the same {n} documents",
        fontsize=14, y=1.00,
    )
    plt.tight_layout()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PNG_OUT, dpi=120, bbox_inches="tight")
    print(f"Wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
