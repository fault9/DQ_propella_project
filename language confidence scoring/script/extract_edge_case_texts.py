import argparse
import csv
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


def edge_cases_for_row(row: dict) -> list[str]:
    cases = []

    if (
        row["content_quality"] in {"excellent", "good"}
        and row["content_safety"] == "safe"
        and row["pii_presence"] == "no_pii"
        and ((not row["language_match"]) or row["low_language_confidence"])
    ):
        cases.append("good_quality_low_language_confidence")

    if (
        row["educational_value"] in {"high", "moderate"}
        and row["content_quality"] in {"poor", "unacceptable"}
        and row["content_safety"] == "safe"
        and row["pii_presence"] == "no_pii"
    ):
        cases.append("educational_low_content_quality")

    if (
        row["information_density"] == "empty"
        and row["content_length"] in {"moderate", "substantial"}
    ):
        cases.append("empty_density_good_length")

    if (
        row["information_density"] == "dense"
        and row["educational_value"] in {"none", "minimal"}
        and row["content_quality"] in {"excellent", "good"}
        and row["content_safety"] == "safe"
        and row["pii_presence"] == "no_pii"
    ):
        cases.append("dense_not_educational")

    if (
        row["content_integrity"] == "complete"
        and row["content_ratio"] in {"complete_content", "mostly_content"}
        and row["information_density"] in {"thin", "empty"}
        and row["educational_value"] in {"none", "minimal"}
        and row["content_safety"] == "safe"
        and row["pii_presence"] == "no_pii"
    ):
        cases.append("complete_but_thin")

    if (
        row["content_integrity"] in {"fragment", "severely_degraded"}
        and row["content_quality"] in {"excellent", "good"}
        and row["information_density"] in {"dense", "adequate"}
        and row["content_safety"] == "safe"
        and row["pii_presence"] == "no_pii"
    ):
        cases.append("fragment_high_quality")

    return cases


OUTPUT_COLUMNS = [
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


def normalize_lid_score(value) -> float | None:
    score = pd.to_numeric(value, errors="coerce")
    if pd.isna(score):
        return None
    return float(max(0, min(1, score)))


def incremental_extract(args) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    propella_ds = load_dataset(
        "openeurollm/propella-annotations",
        "finepdfs",
        split=args.language,
        streaming=True,
    )
    finepdfs_ds = load_dataset(
        "HuggingFaceFW/finepdfs",
        args.language,
        split="train",
        streaming=True,
    )

    propella_by_id = {}
    for row in islice(propella_ds, args.propella_scan):
        propella_by_id[row["id"]] = {
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
        }

    counts = {edge_case: 0 for edge_case in EDGE_CASES}
    seen = set()
    scanned = 0

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        try:
            for finepdf_row in islice(finepdfs_ds, args.finepdfs_scan):
                scanned += 1
                if scanned % args.progress_every == 0:
                    print(f"Scanned {scanned} FinePDFs rows. Current counts: {counts}", flush=True)

                propella_row = propella_by_id.get(finepdf_row["id"])
                if propella_row is None:
                    continue

                full_doc_lid_score = normalize_lid_score(finepdf_row["full_doc_lid_score"])
                language_match = finepdf_row["language"] == finepdf_row["full_doc_lid"]
                low_language_confidence = full_doc_lid_score is None or full_doc_lid_score < 0.70

                combined = {
                    **propella_row,
                    "language": finepdf_row["language"],
                    "raw_text_excerpt": str(finepdf_row["text"])[:args.text_chars],
                    "full_doc_lid": finepdf_row["full_doc_lid"],
                    "full_doc_lid_score": full_doc_lid_score,
                    "language_match": language_match,
                    "low_language_confidence": low_language_confidence,
                    "token_count": finepdf_row["token_count"],
                    "url": finepdf_row["url"],
                }

                for edge_case in edge_cases_for_row(combined):
                    key = (edge_case, combined["id"])
                    if counts[edge_case] >= args.per_case or key in seen:
                        continue

                    writer.writerow({
                        "edge_case": edge_case,
                        **{column: combined.get(column) for column in OUTPUT_COLUMNS if column != "edge_case"},
                    })
                    file.flush()
                    seen.add(key)
                    counts[edge_case] += 1
                    print(f"Found {edge_case}: {counts[edge_case]}/{args.per_case}", flush=True)

                if all(count >= args.per_case for count in counts.values()):
                    print("Found enough examples for all edge cases.", flush=True)
                    break
        except KeyboardInterrupt:
            print("\nInterrupted. Partial results have already been saved.", flush=True)

    print("\nFinal counts:")
    print(counts)
    print(f"Saved partial/final output: {output_path}")


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
    parser.add_argument("--progress-every", type=int, default=5_000)
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use the older batch merge path instead of incremental writes.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/edge_case_texts_swe.csv",
    )
    args = parser.parse_args()

    if not args.batch:
        incremental_extract(args)
        return

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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample[OUTPUT_COLUMNS].to_csv(output_path, index=False)

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
