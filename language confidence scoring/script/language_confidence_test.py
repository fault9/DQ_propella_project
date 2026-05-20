"""
1. Load Propella annotations for one language.
2. Load FinePDFs for the same language.
3. Match rows by id.
4. Extract FinePDFs language-confidence fields.
5. Add derived flags:
   - language_match
   - low_confidence
6. Save CSV/Parquet.
"""

import argparse
import sys
from itertools import islice
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def load_propella_rows(language: str, n_rows: int) -> pd.DataFrame:
    ds = load_dataset(
        "openeurollm/propella-annotations",
        "finepdfs",
        split=language,
        streaming=True,
    )

    rows = []
    for row in islice(ds, n_rows):
        rows.append({
            "id": row["id"],
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


def load_finepdf_rows(language: str, target_ids: set, max_scan: int = 50_000) -> pd.DataFrame:
    ds = load_dataset(
        "HuggingFaceFW/finepdfs",
        language,
        split="train",
        streaming=True,
    )

    rows = []
    remaining = set(target_ids)
    for row in islice(ds, max_scan):
        if row["id"] in remaining:
            rows.append({
                "id": row["id"],
                "language": row["language"],
                "full_doc_lid": row["full_doc_lid"],
                "full_doc_lid_score": row["full_doc_lid_score"],
                "page_average_lid": row["page_average_lid"],
                "page_average_lid_score": row["page_average_lid_score"],
                "token_count": row["token_count"],
                "minhash_cluster_size": row["minhash_cluster_size"],
                "duplicate_count": row["duplicate_count"],
                "url": row["url"],
            })
            remaining.discard(row["id"])
            if not remaining:
                break

    if remaining:
        print(f"Warning: {len(remaining)} IDs not found within {max_scan} scanned rows.")

    return pd.DataFrame(rows)


def add_language_confidence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["full_doc_lid_score"] = df["full_doc_lid_score"].clip(0, 1)
    df["page_average_lid_score"] = df["page_average_lid_score"].clip(0, 1)

    df["language_match_finepdfs"] = df["language"] == df["full_doc_lid"]
    df["low_confidence_finepdfs"] = df["full_doc_lid_score"] < 0.70

    return df


def print_summary(df: pd.DataFrame) -> None:
    print("Rows:", len(df))

    if len(df) == 0:
        return

    print("Mean full-doc LID score:", df["full_doc_lid_score"].mean())
    print("Median full-doc LID score:", df["full_doc_lid_score"].median())
    print("Language match rate:", df["language_match_finepdfs"].mean())
    print("Low confidence rate:", df["low_confidence_finepdfs"].mean())

    print("\nPredicted languages:")
    print(df["full_doc_lid"].value_counts().head(10))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="swe_Latn")
    parser.add_argument("--n-rows", type=int, default=1000)
    parser.add_argument(
        "--output",
        default="data/processed/language_confidence_finepdfs_swe.csv",
    )
    args = parser.parse_args()

    print(f"Loading Propella rows: {args.language}, n={args.n_rows}")
    propella_df = load_propella_rows(args.language, args.n_rows)

    print(f"Loading FinePDFs rows for {len(propella_df)} Propella IDs...")
    finepdf_df = load_finepdf_rows(args.language, set(propella_df["id"]))

    merged = propella_df.merge(finepdf_df, on="id", how="inner")
    merged = add_language_confidence_features(merged)

    print("Propella rows:", len(propella_df))
    print("FinePDFs rows:", len(finepdf_df))
    print("Merged rows:", len(merged))

    print_summary(merged)

    columns = [
        "id",
        "language",
        "full_doc_lid",
        "full_doc_lid_score",
        "page_average_lid",
        "page_average_lid_score",
        "language_match_finepdfs",
        "low_confidence_finepdfs",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    merged[columns].to_parquet(output_path.with_suffix(".parquet"), index=False)
    print(f"\nSaved: {output_path.with_suffix('.parquet')}")
    sys.exit(0)


if __name__ == "__main__":
    main()
