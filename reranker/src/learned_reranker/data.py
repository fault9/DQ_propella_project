from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from learned_reranker.config import DatasetConfig
from learned_reranker.schema import RAW_FINEPDF_COLUMNS, RAW_PROPELLA_COLUMNS


@dataclass(frozen=True)
class JoinReport:
    propella_rows: int
    finepdf_rows: int
    matched_rows: int
    dropped_unmatched_propella: int
    dropped_unmatched_finepdf: int

    def to_dict(self) -> dict[str, int]:
        return {
            "propella_rows": self.propella_rows,
            "finepdf_rows": self.finepdf_rows,
            "matched_rows": self.matched_rows,
            "dropped_unmatched_propella": self.dropped_unmatched_propella,
            "dropped_unmatched_finepdf": self.dropped_unmatched_finepdf,
        }


def _hf_to_polars(dataset: Any, columns: tuple[str, ...], limit: int | None) -> pl.DataFrame:
    return _hf_to_polars_matching(dataset, columns, limit=limit, target_ids=None)


def canonical_doc_id(value: Any) -> str:
    text = "" if value is None else str(value)
    if text.startswith("<urn:uuid:") and text.endswith(">"):
        return text[len("<urn:uuid:") : -1]
    return text


def _hf_to_polars_matching(
    dataset: Any,
    columns: tuple[str, ...],
    limit: int | None,
    target_ids: set[str] | None,
) -> pl.DataFrame:
    canonical_targets = {canonical_doc_id(value) for value in target_ids} if target_ids else None
    rows = []
    matched: set[str] = set()
    for idx, row in enumerate(dataset):
        if canonical_targets is None and limit is not None and idx >= limit:
            break
        row_id = canonical_doc_id(row.get("id"))
        if canonical_targets is not None:
            if row_id not in canonical_targets:
                continue
            matched.add(row_id)
        rows.append({column: row.get(column) for column in columns})
        if canonical_targets is not None and matched == canonical_targets:
            break
    if not rows:
        return pl.DataFrame({column: [] for column in columns})
    return pl.DataFrame(rows)


def _load_dataset_flexible(path: str, config: str, split: str) -> Any:
    from datasets import load_dataset

    try:
        return load_dataset(path, config, split=split, streaming=True)
    except Exception as first_error:
        try:
            return load_dataset(path, split=split, streaming=True)
        except Exception as second_error:
            raise RuntimeError(
                f"Could not load Hugging Face dataset {path!r} with config {config!r} "
                f"or without a config. First error: {first_error}. "
                f"Fallback error: {second_error}."
            ) from second_error


def load_huggingface_frames(
    config: DatasetConfig,
    target_ids: set[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if config.propella_source_dataset:
        propella = _load_dataset_flexible(
            config.propella_dataset, config.propella_source_dataset, config.propella_config
        )
    else:
        propella = _load_dataset_flexible(
            config.propella_dataset, config.propella_config, config.split
        )
    finepdf = _load_dataset_flexible(config.finepdf_dataset, config.finepdf_config, config.split)
    propella_frame = _hf_to_polars_matching(
        propella, RAW_PROPELLA_COLUMNS, limit=config.limit, target_ids=target_ids
    )
    if config.propella_source_dataset and "source_dataset" in propella_frame.columns:
        propella_frame = propella_frame.filter(
            pl.col("source_dataset") == config.propella_source_dataset
        )
    return (
        propella_frame,
        _hf_to_polars_matching(finepdf, RAW_FINEPDF_COLUMNS, limit=config.limit, target_ids=target_ids),
    )


def join_propella_finepdf(
    propella: pl.DataFrame,
    finepdf: pl.DataFrame,
) -> tuple[pl.DataFrame, JoinReport]:
    propella_ids = set(propella["id"].to_list()) if "id" in propella.columns else set()
    finepdf_ids = set(finepdf["id"].to_list()) if "id" in finepdf.columns else set()
    matched_ids = propella_ids & finepdf_ids
    joined = propella.join(finepdf, on="id", how="inner", suffix="_finepdf")
    report = JoinReport(
        propella_rows=propella.height,
        finepdf_rows=finepdf.height,
        matched_rows=joined.height,
        dropped_unmatched_propella=len(propella_ids - matched_ids),
        dropped_unmatched_finepdf=len(finepdf_ids - matched_ids),
    )
    return joined, report
