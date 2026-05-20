import argparse
from itertools import islice
from pathlib import Path

import pandas as pd
from datasets import load_dataset


EDGE_CASES = {
    "good_quality_low_language_confidence": {
        "description": "Good/excellent safe documents whose FinePDFs language ID is low confidence or not the expected language.",
    },
    "educational_low_content_quality": {
        "description": "Educational documents that Propella marks as poor/unacceptable quality.",
    },
    "empty_density_good_length": {
        "description": "Documents with empty information density but moderate/substantial length.",
    },
    "dense_not_educational": {
        "description": "Dense, good/excellent documents with little or no educational value.",
    },
    "complete_but_thin": {
        "description": "Complete documents with mostly/complete content but thin or empty information density.",
    },
    "fragment_high_quality": {
        "description": "Fragmented or degraded documents that Propella still marks as good/excellent quality.",
    },
}


def load_propella_candidates(language: str, max_scan: int) -> pd.DataFrame:
    ds = load_dataset(
        "openeurollm/propella-annotations",
        "finepdfs",
        split=language,
        streaming=True,
    )

    rows = []
    for row in islice(ds, max_scan):
        rows.append({
            "id": row["id"],
            "one_sentence_description": row["one_sentence_description"],
            "content_quality": row["content_quality"],
            "information_density": row["information_density"],
            "educational_value": row["educational_value"],
            "content_safety": row["content_safety"],
            "pii_presence": row["pii_presence"],
            "content_integrity": row["content_integrity"],
            "content_ratio": row["content_ratio"],
            "content_length": row["content_length"],
        })

    return pd.DataFrame(rows)


def load_finepdf_rows(language: str, target_ids: set[str], max_scan: int) -> pd.DataFrame:
    ds = load_dataset(
        "HuggingFaceFW/finepdfs",
        language,
        split="train",
        streaming=True,
    )

    rows = []
    remaining = set(target_ids)

    for row in islice(ds, max_scan):
        if row["id"] not in remaining:
            continue

        rows.append({
            "id": row["id"],
            "raw_text": row["text"],
            "language": row["language"],
            "full_doc_lid": row["full_doc_lid"],
            "full_doc_lid_score": row["full_doc_lid_score"],
            "page_average_lid": row["page_average_lid"],
            "page_average_lid_score": row["page_average_lid_score"],
            "token_count": row["token_count"],
            "url": row["url"],
        })
        remaining.discard(row["id"])

        if not remaining:
            break

    if remaining:
        print(f"Warning: {len(remaining)} target IDs not found within {max_scan} FinePDFs rows.")

    return pd.DataFrame(rows)


def add_language_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["full_doc_lid_score"] = pd.to_numeric(df["full_doc_lid_score"], errors="coerce").clip(0, 1)
    df["page_average_lid_score"] = pd.to_numeric(df["page_average_lid_score"], errors="coerce").clip(0, 1)
    df["language_match"] = df["language"] == df["full_doc_lid"]
    df["low_language_confidence"] = df["full_doc_lid_score"].isna() | (df["full_doc_lid_score"] < 0.70)
    return df


def label_edge_cases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cases = []

    cases.append(df[
        df["content_quality"].isin(["excellent", "good"])
        & (df["content_safety"] == "safe")
        & (df["pii_presence"] == "no_pii")
        & ((~df["language_match"]) | df["low_language_confidence"])
    ].assign(edge_case="good_quality_low_language_confidence"))

    cases.append(df[
        df["educational_value"].isin(["high", "moderate"])
        & df["content_quality"].isin(["poor", "unacceptable"])
        & (df["content_safety"] == "safe")
        & (df["pii_presence"] == "no_pii")
    ].assign(edge_case="educational_low_content_quality"))

    cases.append(df[
        (df["information_density"] == "empty")
        & df["content_length"].isin(["moderate", "substantial"])
    ].assign(edge_case="empty_density_good_length"))

    cases.append(df[
        (df["information_density"] == "dense")
        & df["educational_value"].isin(["none", "minimal"])
        & df["content_quality"].isin(["excellent", "good"])
        & (df["content_safety"] == "safe")
        & (df["pii_presence"] == "no_pii")
    ].assign(edge_case="dense_not_educational"))

    cases.append(df[
        (df["content_integrity"] == "complete")
        & df["content_ratio"].isin(["complete_content", "mostly_content"])
        & df["information_density"].isin(["thin", "empty"])
        & df["educational_value"].isin(["none", "minimal"])
        & (df["content_safety"] == "safe")
        & (df["pii_presence"] == "no_pii")
    ].assign(edge_case="complete_but_thin"))

    cases.append(df[
        df["content_integrity"].isin(["fragment", "severely_degraded"])
        & df["content_quality"].isin(["excellent", "good"])
        & df["information_density"].isin(["dense", "adequate"])
        & (df["content_safety"] == "safe")
        & (df["pii_presence"] == "no_pii")
    ].assign(edge_case="fragment_high_quality"))

    if not cases:
        return pd.DataFrame()

    return pd.concat(cases, ignore_index=True).drop_duplicates(["edge_case", "id"])


def balanced_sample(df: pd.DataFrame, per_case: int) -> pd.DataFrame:
    if df.empty:
        return df

    return (
        df.groupby("edge_case", group_keys=False)
        .apply(lambda group: group.head(per_case), include_groups=True)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="swe_Latn")
    parser.add_argument("--propella-scan", type=int, default=50_000)
    parser.add_argument("--finepdfs-scan", type=int, default=100_000)
    parser.add_argument("--per-case", type=int, default=2)
    parser.add_argument("--text-chars", type=int, default=5_000)
    parser.add_argument(
        "--output",
        default="data/processed/edge_case_texts_swe.csv",
    )
    args = parser.parse_args()

    print(f"Scanning Propella {args.language}: {args.propella_scan} rows")
    propella_df = load_propella_candidates(args.language, args.propella_scan)

    print(f"Loading FinePDFs rows for {len(propella_df)} Propella IDs")
    finepdf_df = load_finepdf_rows(args.language, set(propella_df["id"]), args.finepdfs_scan)

    merged = propella_df.merge(finepdf_df, on="id", how="inner")
    merged = add_language_flags(merged)

    print("Propella rows:", len(propella_df))
    print("FinePDFs matched rows:", len(finepdf_df))
    print("Merged rows:", len(merged))

    edge_cases = label_edge_cases(merged)
    sample = balanced_sample(edge_cases, args.per_case)
    sample["raw_text_excerpt"] = sample["raw_text"].str.slice(0, args.text_chars)

    columns = [
        "edge_case",
        "id",
        "language",
        "one_sentence_description",
        "raw_text_excerpt",
        "content_quality",
        "information_density",
        "educational_value",
        "content_safety",
        "pii_presence",
        "content_integrity",
        "content_ratio",
        "content_length",
        "full_doc_lid",
        "full_doc_lid_score",
        "language_match",
        "low_language_confidence",
        "token_count",
        "url",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(output_path, index=False)

    print("\nFound edge cases:")
    if edge_cases.empty:
        print("None found. Try increasing --propella-scan and --finepdfs-scan.")
    else:
        print(edge_cases["edge_case"].value_counts())

    print("\nSaved sample:")
    print(sample["edge_case"].value_counts() if not sample.empty else "No rows")
    print(output_path)


if __name__ == "__main__":
    main()
