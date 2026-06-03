"""
score_documents.py — Derive a single comparable Bradley-Terry score per document
from the A/B test pairwise preferences.

The Bradley-Terry model turns pairwise win/loss records into a log-odds score
on a consistent scale: P(doc_i beats doc_j) = 1 / (1 + exp(score_j - score_i)).
A difference of 1.0 in score means the higher-scoring doc wins ~73% of the time.

Outputs: document_scores.csv  (doc_id, score, wins, losses, n_comparisons)

Docs that appeared only in skipped pairs are excluded entirely.
Both rating axes (readability + substance) are combined with equal weight.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR    = Path(__file__).parent
RESPONSES_CSV = SCRIPT_DIR / "responses.csv"
PAIRS_PARQUET = SCRIPT_DIR / "pairs.parquet"
OUTPUT_CSV    = SCRIPT_DIR / "document_scores.csv"

BRADLEY_TERRY_MAX_ITER = 500
BRADLEY_TERRY_TOL      = 1e-9
# Regularisation: add this many pseudo-wins and pseudo-losses to every doc.
# Prevents scores from diverging to ±∞ for docs that never won / never lost.
PRIOR_STRENGTH = 0.5


def chosen_doc(choice: str, side_assignment: str) -> str:
    """Map display choice + side_assignment → original doc label ('a' or 'b')."""
    chose_a_display = str(choice).startswith("a_")
    if side_assignment == "a_left":
        return "a" if chose_a_display else "b"
    else:  # a_right: doc_b is shown as display-A
        return "b" if chose_a_display else "a"


def bradley_terry(wins: dict, comparisons: dict) -> dict:
    """
    Iterative MLE for the Bradley-Terry model (Zermelo's algorithm).

    wins[i]            = total wins for doc i across all comparisons
    comparisons[i][j]  = number of times i and j were compared (symmetric: set both)

    Returns dict: doc_id -> log-odds score (zero-meaned).
    """
    docs = list(wins.keys())
    n    = len(docs)

    # Convert to arrays
    w = np.array([wins[d] for d in docs], dtype=float)
    C = np.zeros((n, n))
    for i, di in enumerate(docs):
        for j, dj in enumerate(docs):
            C[i, j] = comparisons[di].get(dj, 0)

    # Initial scores
    s = np.ones(n)

    for _ in range(BRADLEY_TERRY_MAX_ITER):
        s_new = np.zeros(n)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if i == j:
                    continue
                n_ij = C[i, j] + C[j, i]
                if n_ij == 0:
                    continue
                denom += n_ij / (s[i] + s[j])
            s_new[i] = w[i] / denom if denom > 0 else s[i]

        # Normalise to geometric mean = 1
        log_s = np.log(np.where(s_new > 0, s_new, 1e-12))
        s_new = np.exp(log_s - log_s.mean())

        if np.max(np.abs(s_new - s)) < BRADLEY_TERRY_TOL:
            s = s_new
            break
        s = s_new

    # Convert to log scale and zero-mean
    log_s = np.log(s)
    log_s -= log_s.mean()

    return {docs[i]: float(log_s[i]) for i in range(n)}


def main():
    resp     = pd.read_csv(RESPONSES_CSV)
    pairs_df = pd.read_parquet(PAIRS_PARQUET)
    pairs    = {row["pair_id"]: row for _, row in pairs_df.iterrows()}

    # Only real pairs that were not skipped
    rated = resp[(resp["pair_kind"] == "real") & (resp["skipped"] == 0)].copy()
    print(f"Non-skipped real responses: {len(rated)}")

    # Accumulate wins and comparison counts per doc
    # Each response contributes two outcomes: one for readability, one for substance
    wins        = defaultdict(int)
    losses      = defaultdict(int)
    comparisons = defaultdict(lambda: defaultdict(int))

    outcomes_used = 0

    for _, row in rated.iterrows():
        pair = pairs.get(row["pair_id"])
        if pair is None:
            continue

        doc_a_id = str(pair["doc_a_id"])
        doc_b_id = str(pair["doc_b_id"])
        sa       = row["side_assignment"]

        for choice_col in ("readability_choice", "substance_choice"):
            choice = row[choice_col]
            if pd.isna(choice):
                continue

            winner_label = chosen_doc(str(choice), sa)
            if winner_label == "a":
                winner, loser = doc_a_id, doc_b_id
            else:
                winner, loser = doc_b_id, doc_a_id

            wins[winner]   += 1
            losses[loser]  += 1
            comparisons[winner][loser] += 1
            comparisons[loser][winner] += 1
            outcomes_used += 1

    print(f"Outcomes used (2 per response × non-null choices): {outcomes_used}")
    print(f"Distinct docs with at least one outcome: {len(wins)}")

    # Docs that only lost and never won still need an entry
    all_docs = set(wins.keys()) | set(losses.keys())
    for d in all_docs:
        wins.setdefault(d, 0)

    # Warn about docs with zero wins (will get very low BT score)
    zero_win = [d for d in all_docs if wins[d] == 0]
    if zero_win:
        print(f"  Note: {len(zero_win)} doc(s) never won a comparison — "
              f"they receive the minimum score.")

    # Add regularisation: PRIOR_STRENGTH pseudo-wins and pseudo-losses per doc,
    # paired against a virtual "ghost" doc with fixed score 0 (geometric mean).
    # This prevents ±∞ scores for docs with perfect win/loss records.
    ghost = "__prior__"
    wins_reg        = dict(wins)
    comparisons_reg = {d: dict(comparisons[d]) for d in all_docs}
    comparisons_reg[ghost] = {}
    wins_reg[ghost] = 0
    for d in all_docs:
        wins_reg[d] = wins_reg.get(d, 0) + PRIOR_STRENGTH
        wins_reg[ghost] += PRIOR_STRENGTH
        comparisons_reg[d][ghost] = comparisons_reg[d].get(ghost, 0) + 1
        comparisons_reg[ghost][d] = comparisons_reg[ghost].get(d, 0) + 1

    # Fit Bradley-Terry
    scores_all = bradley_terry(wins_reg, comparisons_reg)
    # Drop the ghost doc; re-zero-mean over real docs only
    scores = {d: scores_all[d] for d in all_docs}
    mean_s = sum(scores.values()) / len(scores)
    scores = {d: v - mean_s for d, v in scores.items()}

    # Build output
    rows = []
    for doc_id in sorted(all_docs):
        w = wins[doc_id]
        l = losses[doc_id]
        n = w + l
        rows.append({
            "doc_id":         doc_id,
            "score":          round(scores[doc_id], 4),
            "wins":           w,
            "losses":         l,
            "n_comparisons":  n,
        })

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nScores written to {OUTPUT_CSV}")
    print(f"\nTop 10 documents:")
    print(out.head(10).to_string(index=False))
    print(f"\nBottom 10 documents:")
    print(out.tail(10).to_string(index=False))
    print(f"\nScore range: {out['score'].min():.3f} to {out['score'].max():.3f}")
    print(f"Median: {out['score'].median():.3f}")
    print("\nInterpretation:")
    print("  score difference of 1.0 → ~73% win probability for the higher-scoring doc")
    print("  score difference of 2.0 → ~88% win probability")
    print("  Use: P(A beats B) = 1 / (1 + exp(score_B - score_A))")


if __name__ == "__main__":
    main()
