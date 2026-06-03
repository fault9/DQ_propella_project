"""
generate_pairs.py
Reads sample.parquet → writes pairs.parquet.
Idempotent: if pairs.parquet exists, prints message and exits 0.
"""
import sys
import math
import random
import uuid
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
SAMPLE_PATH = SCRIPT_DIR / "sample.parquet"
PAIRS_PATH = SCRIPT_DIR / "pairs.parquet"

SEED = 42

# ---------------------------------------------------------------------------
# Topic family mapping
# ---------------------------------------------------------------------------
FAMILY_MAP = {
    "news_report": "informational",
    "reference": "informational",
    "technical_documentation": "informational",
    "analytical": "informational",
    "structured_data": "informational",
    "qa_structured": "informational",
    "instructional": "informational",
    "procedural": "informational",
    "legal_document": "informational",
    "specification_standard": "informational",
    "transactional": "transactional",
    "press_release": "transactional",
    "boilerplate": "transactional",
    "creative": "creative_conversational",
    "conversational": "creative_conversational",
    "opinion_editorial": "creative_conversational",
    "review_critique": "creative_conversational",
}

LENGTH_BUCKET = {
    "minimal": 0,
    "brief": 1,
    "moderate": 2,
    "substantial": 3,
}

# Annotation fields (all except id, text, url, char_count, one_sentence_description, content_length)
ANNOTATION_FIELDS = [
    "content_integrity",
    "content_ratio",
    "content_type",
    "business_sector",
    "technical_content",
    "information_density",
    "content_quality",
    "audience_level",
    "commercial_bias",
    "time_sensitivity",
    "content_safety",
    "educational_value",
    "reasoning_indicators",
    "pii_presence",
    "regional_relevance",
    "country_relevance",
]


def to_python(val):
    """Convert numpy scalars/arrays to Python native types."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def get_first_content_type(val):
    if isinstance(val, np.ndarray) and len(val) > 0:
        return str(val[0])
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return str(val[0])
    return str(val)


def assign_family(ct_first):
    return FAMILY_MAP.get(ct_first, "informational")


def main():
    if PAIRS_PATH.exists():
        print(f"pairs.parquet already exists at {PAIRS_PATH}. Delete it to regenerate.")
        sys.exit(0)

    df = pd.read_parquet(SAMPLE_PATH)
    print(f"Loaded {len(df)} docs from {SAMPLE_PATH}")

    # Assign topic family
    df["_ct_first"] = df["content_type"].apply(get_first_content_type)
    df["topic_family"] = df["_ct_first"].apply(assign_family)

    # Length bucket
    df["_length_bucket"] = df["content_length"].map(LENGTH_BUCKET).fillna(0).astype(int)

    # Per-family doc count
    family_counts = df["topic_family"].value_counts()
    print("\nDocs per topic family:")
    for fam, cnt in family_counts.items():
        print(f"  {fam}: {cnt}")

    families = df["topic_family"].unique().tolist()
    total_family_docs = {f: len(df[df["topic_family"] == f]) for f in families}

    # ---------------------------------------------------------------------------
    # Generate 99 real pairs
    # ---------------------------------------------------------------------------
    rng = random.Random(SEED)
    # Cap: each doc appears at most 5 pairs
    doc_appearance = {doc_id: 0 for doc_id in df["id"]}

    # Proportional targets per family
    total_target = 99
    family_target = {}
    total_docs = len(df)
    for fam in families:
        family_target[fam] = max(1, round(total_family_docs[fam] / total_docs * total_target))

    # Adjust to sum exactly to 99
    diff = sum(family_target.values()) - total_target
    sorted_fams = sorted(families, key=lambda f: -family_target[f])
    for i in range(abs(diff)):
        if diff > 0:
            family_target[sorted_fams[i % len(sorted_fams)]] -= 1
        else:
            family_target[sorted_fams[i % len(sorted_fams)]] += 1

    print(f"\nTarget pairs per family: {family_target}")

    real_pairs = []
    used_pairs_set = set()

    def can_use(doc_id, cap=5):
        return doc_appearance[doc_id] < cap

    def try_generate_pairs_for_family(fam, target):
        fam_docs = df[df["topic_family"] == fam].copy()
        indices = fam_docs.index.tolist()
        rng.shuffle(indices)
        generated = []
        attempts = 0
        max_attempts = target * 500
        while len(generated) < target and attempts < max_attempts:
            attempts += 1
            i, j = rng.sample(indices, 2)
            id_a = df.loc[i, "id"]
            id_b = df.loc[j, "id"]
            # Canonical pair key
            key = tuple(sorted([id_a, id_b]))
            if key in used_pairs_set:
                continue
            if not (can_use(id_a) and can_use(id_b)):
                continue
            bucket_a = df.loc[i, "_length_bucket"]
            bucket_b = df.loc[j, "_length_bucket"]
            if abs(int(bucket_a) - int(bucket_b)) > 1:
                continue
            generated.append((i, j))
            used_pairs_set.add(key)
            doc_appearance[id_a] += 1
            doc_appearance[id_b] += 1
        return generated, target - len(generated)

    shortfalls = {}
    for fam in families:
        pairs_fam, shortfall = try_generate_pairs_for_family(fam, family_target.get(fam, 0))
        real_pairs.extend(pairs_fam)
        if shortfall > 0:
            shortfalls[fam] = shortfall

    if shortfalls:
        print(f"\nShortfall per family: {shortfalls}")
    else:
        print("\nNo shortfalls — all family targets met.")

    # Trim or extend to exactly 99
    if len(real_pairs) > 99:
        # Remove extras from the end
        for i_idx, j_idx in real_pairs[99:]:
            id_a = df.loc[i_idx, "id"]
            id_b = df.loc[j_idx, "id"]
            key = tuple(sorted([id_a, id_b]))
            used_pairs_set.discard(key)
            doc_appearance[id_a] -= 1
            doc_appearance[id_b] -= 1
        real_pairs = real_pairs[:99]
    elif len(real_pairs) < 99:
        # Try to fill from any family
        needed = 99 - len(real_pairs)
        print(f"\nTrying to fill {needed} remaining pairs from any family...")
        all_indices = df.index.tolist()
        attempts = 0
        max_attempts = needed * 2000
        while len(real_pairs) < 99 and attempts < max_attempts:
            attempts += 1
            i, j = rng.sample(all_indices, 2)
            id_a = df.loc[i, "id"]
            id_b = df.loc[j, "id"]
            key = tuple(sorted([id_a, id_b]))
            if key in used_pairs_set:
                continue
            if not (can_use(id_a) and can_use(id_b)):
                continue
            bucket_a = df.loc[i, "_length_bucket"]
            bucket_b = df.loc[j, "_length_bucket"]
            if abs(int(bucket_a) - int(bucket_b)) > 1:
                continue
            real_pairs.append((i, j))
            used_pairs_set.add(key)
            doc_appearance[id_a] += 1
            doc_appearance[id_b] += 1

    print(f"\nGenerated {len(real_pairs)} real pairs.")

    # ---------------------------------------------------------------------------
    # Mark 9 as duplicates
    # ---------------------------------------------------------------------------
    dup_rng = random.Random(42)
    dup_indices = dup_rng.sample(range(len(real_pairs)), 9)
    dup_set = set(dup_indices)

    # ---------------------------------------------------------------------------
    # Build calibration pairs (9 total)
    # ---------------------------------------------------------------------------
    # 5 readability calibration
    broken_mask = df["content_integrity"].isin(["severely_degraded", "fragment"])
    clean_mask = (
        (df["content_integrity"] == "complete")
        & (df["content_quality"].isin(["good", "excellent"]))
        & (df["information_density"] == "dense")
    )
    broken_docs = df[broken_mask].index.tolist()
    clean_docs = df[clean_mask].index.tolist()

    cal_rng = random.Random(SEED)
    cal_rng.shuffle(broken_docs)
    cal_rng.shuffle(clean_docs)

    readability_cal = []
    used_cal = set()
    # Try same family first
    for b_idx in broken_docs:
        if len(readability_cal) >= 5:
            break
        b_fam = df.loc[b_idx, "topic_family"]
        matched = False
        for c_idx in clean_docs:
            if c_idx in used_cal or b_idx in used_cal:
                continue
            c_fam = df.loc[c_idx, "topic_family"]
            if c_fam == b_fam:
                readability_cal.append((b_idx, c_idx))
                used_cal.add(b_idx)
                used_cal.add(c_idx)
                matched = True
                break
        if not matched:
            for c_idx in clean_docs:
                if c_idx in used_cal or b_idx in used_cal:
                    continue
                readability_cal.append((b_idx, c_idx))
                used_cal.add(b_idx)
                used_cal.add(c_idx)
                break

    # 4 substance calibration
    low_edu_mask = (
        (df["educational_value"].isin(["none", "minimal"]))
        & (df["content_integrity"] == "complete")
    )
    high_edu_mask = (
        (df["educational_value"] == "high")
        & (df["content_integrity"] == "complete")
    )
    low_edu_docs = df[low_edu_mask].index.tolist()
    high_edu_docs = df[high_edu_mask].index.tolist()

    cal_rng2 = random.Random(SEED + 1)
    cal_rng2.shuffle(low_edu_docs)
    cal_rng2.shuffle(high_edu_docs)

    substance_cal = []
    used_cal2 = set()
    for l_idx in low_edu_docs:
        if len(substance_cal) >= 4:
            break
        l_fam = df.loc[l_idx, "topic_family"]
        matched = False
        for h_idx in high_edu_docs:
            if h_idx in used_cal2 or l_idx in used_cal2:
                continue
            h_fam = df.loc[h_idx, "topic_family"]
            if h_fam == l_fam:
                substance_cal.append((l_idx, h_idx))
                used_cal2.add(l_idx)
                used_cal2.add(h_idx)
                matched = True
                break
        if not matched:
            for h_idx in high_edu_docs:
                if h_idx in used_cal2 or l_idx in used_cal2:
                    continue
                substance_cal.append((l_idx, h_idx))
                used_cal2.add(l_idx)
                used_cal2.add(h_idx)
                break

    print(f"\nCalibration pairs: {len(readability_cal)} readability, {len(substance_cal)} substance")

    # ---------------------------------------------------------------------------
    # Build rows for DataFrame
    # ---------------------------------------------------------------------------
    rows = []

    def doc_row(idx, prefix):
        row = df.loc[idx]
        d = {}
        d[f"{prefix}id"] = row["id"]
        d[f"{prefix}text"] = row["text"] if isinstance(row["text"], str) else ""
        d[f"{prefix}description"] = row["one_sentence_description"]
        d[f"{prefix}char_count"] = int(row["char_count"])
        d[f"{prefix}topic_family"] = row["topic_family"]
        d[f"{prefix}content_length"] = row["content_length"]
        d[f"{prefix}url"] = row["url"] if isinstance(row["url"], str) else ""
        for field in ANNOTATION_FIELDS:
            val = row[field]
            d[f"{prefix}{field}"] = to_python(val)
        return d

    # Real pairs
    for rank, (i_idx, j_idx) in enumerate(real_pairs):
        pair_id = str(uuid.uuid4())
        is_dup = rank in dup_set
        r = {"pair_id": pair_id, "pair_kind": "real", "is_duplicate": is_dup}
        r.update(doc_row(i_idx, "doc_a_"))
        r.update(doc_row(j_idx, "doc_b_"))
        r["calibration_correct_doc"] = None
        rows.append(r)

    # Calibration readability
    for b_idx, c_idx in readability_cal:
        pair_id = str(uuid.uuid4())
        r = {"pair_id": pair_id, "pair_kind": "calibration_readability", "is_duplicate": False}
        r.update(doc_row(b_idx, "doc_a_"))
        r.update(doc_row(c_idx, "doc_b_"))
        r["calibration_correct_doc"] = "b"
        rows.append(r)

    # Calibration substance
    for l_idx, h_idx in substance_cal:
        pair_id = str(uuid.uuid4())
        r = {"pair_id": pair_id, "pair_kind": "calibration_substance", "is_duplicate": False}
        r.update(doc_row(l_idx, "doc_a_"))
        r.update(doc_row(h_idx, "doc_b_"))
        r["calibration_correct_doc"] = "b"
        rows.append(r)

    pairs_df = pd.DataFrame(rows)

    # Ensure correct column order
    base_cols = [
        "pair_id", "pair_kind", "is_duplicate",
        "doc_a_id", "doc_b_id",
        "doc_a_text", "doc_b_text",
        "doc_a_description", "doc_b_description",
        "doc_a_char_count", "doc_b_char_count",
        "doc_a_topic_family", "doc_b_topic_family",
        "doc_a_content_length", "doc_b_content_length",
        "doc_a_url", "doc_b_url",
        "calibration_correct_doc",
    ]
    ann_cols = []
    for field in ANNOTATION_FIELDS:
        ann_cols.append(f"doc_a_{field}")
        ann_cols.append(f"doc_b_{field}")

    all_cols = base_cols + ann_cols
    pairs_df = pairs_df[all_cols]

    pairs_df.to_parquet(PAIRS_PATH, index=False)
    print(f"\nWrote {len(pairs_df)} rows to {PAIRS_PATH}")

    # ---------------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------------
    print("\n--- Pairs by kind ---")
    print(pairs_df["pair_kind"].value_counts().to_string())
    print(f"  is_duplicate=True: {pairs_df['is_duplicate'].sum()}")

    real_df = pairs_df[pairs_df["pair_kind"] == "real"]
    print("\n--- Topic family distribution (real pairs) ---")
    fam_counts_a = real_df["doc_a_topic_family"].value_counts()
    fam_counts_b = real_df["doc_b_topic_family"].value_counts()
    combined = (fam_counts_a.add(fam_counts_b, fill_value=0) / 2).sort_values(ascending=False)
    print(combined.to_string())

    print("\n--- Length bucket gap distribution (real pairs) ---")
    real_df = real_df.copy()
    real_df["_gap"] = real_df.apply(
        lambda r: abs(LENGTH_BUCKET.get(r["doc_a_content_length"], 0) - LENGTH_BUCKET.get(r["doc_b_content_length"], 0)),
        axis=1,
    )
    print(real_df["_gap"].value_counts().sort_index().to_string())

    print("\n--- Doc reuse stats ---")
    all_doc_ids = list(real_df["doc_a_id"]) + list(real_df["doc_b_id"])
    from collections import Counter
    counts = Counter(all_doc_ids)
    appearances = list(counts.values())
    appearances.sort()
    median_app = appearances[len(appearances) // 2]
    p90_app = appearances[int(len(appearances) * 0.9)]
    max_app = max(appearances)
    distinct = len(counts)
    print(f"  Distinct docs used: {distinct} of 263")
    print(f"  Median appearances: {median_app}")
    print(f"  P90 appearances:    {p90_app}")
    print(f"  Max appearances:    {max_app}")


if __name__ == "__main__":
    main()
