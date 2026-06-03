"""
analyze_results.py — Interpret A/B test results against propella-1 labels.

Reads responses.csv + pairs.parquet and prints:
  1. Per-rater calibration accuracy
  2. Propella label agreement rates (readability + substance)
  3. Axis separability (% pairs where readability ≠ substance direction)
  4. Inter-rater agreement on duplicate pairs
  5. Timing stats

Also writes agreement_detail.csv for further inspection.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / "hf_cache"))

RESPONSES_CSV  = SCRIPT_DIR / "responses.csv"
PAIRS_PARQUET  = SCRIPT_DIR / "pairs.parquet"
OUTPUT_CSV     = SCRIPT_DIR / "agreement_detail.csv"

# Ordinal label rankings (higher index = better)
QUALITY_RANK = {"unacceptable": 0, "poor": 1, "adequate": 2, "good": 3, "excellent": 4}
EDU_RANK     = {"none": 0, "minimal": 1, "basic": 2, "moderate": 3, "high": 4}


def chosen_doc(choice: str, side_assignment: str) -> str:
    """Map display choice + side assignment → original doc 'a' or 'b'."""
    chose_a_display = str(choice).startswith("a_")
    if side_assignment == "a_left":
        return "a" if chose_a_display else "b"
    else:  # a_right: doc_b is shown as display-A
        return "b" if chose_a_display else "a"


def choice_direction(choice: str) -> str:
    """Return 'a' or 'b' based on which display side the rater favoured."""
    return "a" if str(choice).startswith("a_") else "b"


def agreement(chosen: str, val_a, val_b, rank_map: dict):
    """
    Returns:
      'agree'    — rater chose the doc with the higher propella label
      'disagree' — rater chose the doc with the lower propella label
      'tie'      — both docs have the same label
      'unknown'  — one or both labels missing from rank_map
    """
    ra = rank_map.get(str(val_a))
    rb = rank_map.get(str(val_b))
    if ra is None or rb is None:
        return "unknown"
    if ra == rb:
        return "tie"
    better = "a" if ra > rb else "b"
    return "agree" if chosen == better else "disagree"


def pct(n, d):
    return f"{100*n/d:.1f}%" if d > 0 else "n/a"


def main():
    if not RESPONSES_CSV.exists():
        print(f"ERROR: {RESPONSES_CSV} not found. Export with:\n"
              f"  sqlite3 -header -csv results.db \"SELECT * FROM responses;\" > responses.csv")
        return

    resp = pd.read_csv(RESPONSES_CSV)
    pairs_df = pd.read_parquet(PAIRS_PARQUET)

    # Build pairs lookup
    pairs = {row["pair_id"]: row for _, row in pairs_df.iterrows()}

    print(f"Loaded {len(resp)} responses from {len(resp['rater_name'].unique())} rater(s).")
    print(f"Pair kinds: {resp['pair_kind'].value_counts().to_dict()}")
    print(f"Skipped: {resp['skipped'].sum()}")

    # ── 1. Per-rater calibration accuracy ────────────────────────────────────
    print("\n" + "="*60)
    print("1. CALIBRATION ACCURACY")
    print("="*60)
    print("(Did raters pick the obviously better doc on control pairs?)\n")

    for rater in sorted(resp["rater_name"].unique()):
        r_resp = resp[resp["rater_name"] == rater]
        for kind, axis in [("calibration_readability", "readability_choice"),
                           ("calibration_substance",   "substance_choice")]:
            cal = r_resp[(r_resp["pair_kind"] == kind) & (r_resp["skipped"] == 0)]
            correct = 0
            total = 0
            for _, row in cal.iterrows():
                pair = pairs.get(row["pair_id"])
                if pair is None:
                    continue
                correct_doc = pair.get("calibration_correct_doc")
                choice = row[axis]
                if pd.isna(choice) or not correct_doc:
                    continue
                total += 1
                if chosen_doc(choice, row["side_assignment"]) == correct_doc:
                    correct += 1
            label = "readability" if "readability" in kind else "substance"
            print(f"  {rater:<20} {label:<12}: {correct}/{total} = {pct(correct, total)}")

    # ── 2. Propella label agreement (real pairs only) ─────────────────────────
    print("\n" + "="*60)
    print("2. PROPELLA LABEL AGREEMENT  (real pairs, non-skipped)")
    print("="*60)
    print("(Did human preferences align with propella's content_quality /")
    print(" educational_value rankings?)\n")

    real = resp[(resp["pair_kind"] == "real") & (resp["skipped"] == 0)].copy()

    detail_rows = []

    for _, row in real.iterrows():
        pair = pairs.get(row["pair_id"])
        if pair is None:
            continue

        qa = pair.get("doc_a_content_quality")
        qb = pair.get("doc_b_content_quality")
        ea = pair.get("doc_a_educational_value")
        eb = pair.get("doc_b_educational_value")

        rc = row["readability_choice"]
        sc = row["substance_choice"]
        sa = row["side_assignment"]

        read_doc = chosen_doc(rc, sa) if not pd.isna(rc) else None
        sub_doc  = chosen_doc(sc, sa) if not pd.isna(sc) else None

        read_agr = agreement(read_doc, qa, qb, QUALITY_RANK) if read_doc else "missing"
        sub_agr  = agreement(sub_doc,  ea, eb, EDU_RANK)     if sub_doc  else "missing"

        # Axis divergence: did rater pick different sides on the two questions?
        read_dir = choice_direction(rc) if not pd.isna(rc) else None
        sub_dir  = choice_direction(sc) if not pd.isna(sc) else None
        diverged = (read_dir is not None and sub_dir is not None
                    and read_dir != sub_dir)

        detail_rows.append({
            "rater_name":          row["rater_name"],
            "pair_id":             row["pair_id"],
            "is_duplicate":        row["is_duplicate"],
            "readability_choice":  rc,
            "substance_choice":    sc,
            "side_assignment":     sa,
            "chosen_read_doc":     read_doc,
            "chosen_sub_doc":      sub_doc,
            "doc_a_quality":       qa,
            "doc_b_quality":       qb,
            "doc_a_edu":           ea,
            "doc_b_edu":           eb,
            "readability_agr":     read_agr,
            "substance_agr":       sub_agr,
            "axes_diverged":       diverged,
        })

    detail_df = pd.DataFrame(detail_rows)

    # Overall agreement
    for axis, col in [("Readability vs content_quality", "readability_agr"),
                      ("Substance vs educational_value",  "substance_agr")]:
        counts = detail_df[col].value_counts()
        agree    = counts.get("agree",    0)
        disagree = counts.get("disagree", 0)
        tie      = counts.get("tie",      0)
        unknown  = counts.get("unknown",  0) + counts.get("missing", 0)
        decisive = agree + disagree  # exclude ties and unknowns
        print(f"  {axis}")
        print(f"    agree={agree}, disagree={disagree}, tie={tie}, unknown={unknown}")
        print(f"    Agreement rate (excl. ties): {pct(agree, decisive)}")
        print()

    # Per-rater breakdown
    print("  Per-rater agreement rates (excl. ties):")
    for rater in sorted(detail_df["rater_name"].unique()):
        sub = detail_df[detail_df["rater_name"] == rater]
        for axis, col in [("readability", "readability_agr"),
                          ("substance",   "substance_agr")]:
            agree    = (sub[col] == "agree").sum()
            disagree = (sub[col] == "disagree").sum()
            decisive = agree + disagree
            print(f"    {rater:<20} {axis:<12}: {pct(agree, decisive)} "
                  f"({agree}/{decisive} decisive pairs)")
    print()

    # ── 3. Axis separability ─────────────────────────────────────────────────
    print("="*60)
    print("3. AXIS SEPARABILITY")
    print("="*60)
    print("(% pairs where readability and substance choices pointed")
    print(" in opposite directions — high % means axes are independent)\n")

    both = detail_df[detail_df["chosen_read_doc"].notna() &
                     detail_df["chosen_sub_doc"].notna()]
    diverged_n = both["axes_diverged"].sum()
    print(f"  Overall: {diverged_n}/{len(both)} pairs = {pct(diverged_n, len(both))}")

    for rater in sorted(both["rater_name"].unique()):
        sub = both[both["rater_name"] == rater]
        n = sub["axes_diverged"].sum()
        print(f"  {rater:<20}: {n}/{len(sub)} = {pct(n, len(sub))}")
    print()

    # ── 4. Inter-rater agreement on duplicates ───────────────────────────────
    print("="*60)
    print("4. INTER-RATER AGREEMENT  (duplicate pairs)")
    print("="*60)
    print("(Pairs seen by 2 raters — do they agree?)\n")

    dup_resp = real[real["is_duplicate"] == 1]
    dup_pairs = dup_resp["pair_id"].unique()
    print(f"  Duplicate pairs with ≥2 ratings: "
          f"{sum(1 for p in dup_pairs if (dup_resp['pair_id']==p).sum() >= 2)}"
          f" / {len(dup_pairs)}")

    exact_read = exact_sub = binary_read = binary_sub = total_dup = 0

    for pid in dup_pairs:
        ratings = dup_resp[dup_resp["pair_id"] == pid]
        if len(ratings) < 2:
            continue
        pairs_of_raters = [(ratings.iloc[i], ratings.iloc[j])
                           for i in range(len(ratings))
                           for j in range(i+1, len(ratings))]
        for r1, r2 in pairs_of_raters:
            rc1, rc2 = r1["readability_choice"], r2["readability_choice"]
            sc1, sc2 = r1["substance_choice"],   r2["substance_choice"]
            if not pd.isna(rc1) and not pd.isna(rc2):
                total_dup += 1
                # Normalize to original doc direction
                d1_r = chosen_doc(rc1, r1["side_assignment"])
                d2_r = chosen_doc(rc2, r2["side_assignment"])
                d1_s = chosen_doc(sc1, r1["side_assignment"]) if not pd.isna(sc1) else None
                d2_s = chosen_doc(sc2, r2["side_assignment"]) if not pd.isna(sc2) else None
                # Exact (same 4-point label, normalized to original doc)
                if rc1 == rc2 and r1["side_assignment"] == r2["side_assignment"]:
                    exact_read += 1
                elif (chosen_doc(rc1, r1["side_assignment"]) ==
                      chosen_doc(rc2, r2["side_assignment"]) and
                      rc1.split("_")[1] == rc2.split("_")[1]):
                    exact_read += 1
                # Binary (just which doc was preferred)
                if d1_r == d2_r:
                    binary_read += 1
                if d1_s and d2_s:
                    if d1_s == d2_s:
                        binary_sub += 1
                    if sc1 == sc2 and r1["side_assignment"] == r2["side_assignment"]:
                        exact_sub += 1
                    elif (chosen_doc(sc1, r1["side_assignment"]) ==
                          chosen_doc(sc2, r2["side_assignment"]) and
                          sc1.split("_")[1] == sc2.split("_")[1]):
                        exact_sub += 1

    print(f"  Readability — binary agreement: {pct(binary_read, total_dup)} "
          f"({binary_read}/{total_dup})")
    print(f"  Readability — exact agreement:  {pct(exact_read,  total_dup)}")
    print(f"  Substance   — binary agreement: {pct(binary_sub,  total_dup)}")
    print(f"  Substance   — exact agreement:  {pct(exact_sub,   total_dup)}")
    print()
    print("  Interpretation:")
    print("    Binary ≥70% → raters reliably agree on which doc is better")
    print("    Exact  ≥50% → raters agree on the strength of preference too")
    print()

    # ── 5. Timing ─────────────────────────────────────────────────────────────
    print("="*60)
    print("5. TIMING")
    print("="*60)
    timed = resp[(resp["skipped"] == 0) & resp["time_ms"].notna()]
    for rater in sorted(timed["rater_name"].unique()):
        sub = timed[timed["rater_name"] == rater]["time_ms"]
        print(f"  {rater:<20}: median {sub.median()/1000:.1f}s, "
              f"p90 {sub.quantile(0.9)/1000:.1f}s, "
              f"min {sub.min()/1000:.1f}s, max {sub.max()/1000:.1f}s")
    print()

    # ── Save detail CSV ───────────────────────────────────────────────────────
    detail_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Per-pair detail written to {OUTPUT_CSV}")
    print("Open it in Excel or Python to inspect individual disagreements.")


if __name__ == "__main__":
    main()
