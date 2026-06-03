"""Merge LLM_train*.csv files into traindata/LLM_train_extended.csv."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from learned_reranker.data import read_local_table

TRAIN_DIR = Path("traindata")
BASE_PATH = TRAIN_DIR / "LLM_train.csv"
TOP_PATH = TRAIN_DIR / "LLM_train_extreme_top.csv"
BOTTOM_PATH = TRAIN_DIR / "LLM_train_extreme_bottom.csv"
OUTPUT_PATH = TRAIN_DIR / "LLM_train_extended.csv"


def _align_to_base_schema(base: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    aligned = frame.select(base.columns)
    casts: list[pl.Expr] = []
    for column, dtype in base.schema.items():
        if aligned.schema[column] != dtype:
            casts.append(pl.col(column).cast(dtype, strict=False))
    if casts:
        aligned = aligned.with_columns(casts)
    return aligned


def merge_llm_train_extended() -> pl.DataFrame:
    base = read_local_table(BASE_PATH)
    base_columns = base.columns
    frames = [
        base,
        _align_to_base_schema(base, read_local_table(TOP_PATH)),
        _align_to_base_schema(base, read_local_table(BOTTOM_PATH)),
    ]
    merged = pl.concat(frames, how="vertical")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.write_csv(OUTPUT_PATH)
    return merged


def main() -> None:
    merged = merge_llm_train_extended()
    print(f"Wrote {merged.height} rows to {OUTPUT_PATH}")
    print(f"Columns ({len(merged.columns)}): {', '.join(merged.columns)}")


if __name__ == "__main__":
    main()
