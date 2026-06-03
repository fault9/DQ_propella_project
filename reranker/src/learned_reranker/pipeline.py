from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from learned_reranker.artifacts import read_pickle, write_json
from learned_reranker.config import L2Mode, OutputFormat, PipelineConfig
from learned_reranker.data import (
    JoinReport,
    canonical_doc_id,
    join_propella_finepdf,
    load_huggingface_frames,
    read_local_table,
)
from learned_reranker.features import (
    add_l2_dedup_features,
    apply_hard_filter,
    apply_percentile_normalization,
    convert_raw_features,
    feature_matrix,
    fit_normalization_stats,
    priority_vector,
)
from learned_reranker.l1 import L1Model, prototype_l1_model
from learned_reranker.l2 import (
    add_interaction_features,
    l2_feature_matrix,
    prototype_l2_score,
)
from learned_reranker.schema import L1_FEATURES, OUTPUT_COLUMNS


@dataclass(frozen=True)
class PipelineResult:
    output: pl.DataFrame
    report: dict[str, Any]


def _load_l1_model(config: PipelineConfig) -> L1Model:
    if config.l1_model_path:
        return read_pickle(config.l1_model_path)
    return prototype_l1_model(config.prototype)


def _load_l2_model(config: PipelineConfig) -> Any | None:
    if config.ranking.l2_mode == L2Mode.trained:
        if not config.l2_model_path:
            raise ValueError("l2_mode=trained requires l2_model_path")
        return read_pickle(config.l2_model_path)
    return None


def _load_local_combined(path: str, target_ids: set[str] | None) -> tuple[pl.DataFrame, JoinReport]:
    """Load a pre-joined combined CSV/Parquet file directly, bypassing HuggingFace."""
    from pathlib import Path as _Path
    p = _Path(path)
    df = read_local_table(p)
    # Normalise the id column so downstream code always sees 'id'
    if "id" not in df.columns and "doc_id" in df.columns:
        df = df.with_columns(pl.col("doc_id").alias("id"))
    if target_ids is not None:
        df = df.filter(
            pl.col("id").map_elements(canonical_doc_id, return_dtype=pl.Utf8).is_in(
                {canonical_doc_id(v) for v in target_ids}
            )
        )
    report = JoinReport(
        propella_rows=df.height,
        finepdf_rows=df.height,
        matched_rows=df.height,
        dropped_unmatched_propella=0,
        dropped_unmatched_finepdf=0,
    )
    return df, report


def prepare_joined_frame(
    config: PipelineConfig,
    propella: pl.DataFrame | None = None,
    finepdf: pl.DataFrame | None = None,
    target_ids: set[str] | None = None,
) -> tuple[pl.DataFrame, JoinReport]:
    if propella is None or finepdf is None:
        from pathlib import Path as _Path
        p = _Path(config.dataset.propella_dataset)
        if p.suffix in (".csv", ".parquet") and p.is_file():
            return _load_local_combined(str(p), target_ids)
        propella, finepdf = load_huggingface_frames(config.dataset, target_ids=target_ids)
    joined, report = join_propella_finepdf(propella, finepdf)
    return joined, report


def prepare_features(raw: pl.DataFrame, config: PipelineConfig) -> tuple[pl.DataFrame, dict[str, Any]]:
    converted = convert_raw_features(raw)
    filtered = apply_hard_filter(converted, config.hard_filter)
    l1_stats = fit_normalization_stats(filtered.filter(pl.col("passed_hard_filter")), L1_FEATURES)
    normalized = apply_percentile_normalization(filtered, l1_stats, L1_FEATURES)
    deduped = add_l2_dedup_features(normalized)
    return deduped, {"l1_normalization": l1_stats.to_dict()}


def _rank_l1(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    model = _load_l1_model(config)
    unit_u = config.priority.as_unit_vector()
    u = priority_vector(unit_u)
    survivors = df.filter(pl.col("passed_hard_filter"))
    if survivors.is_empty():
        return df.with_columns(
            [
                pl.lit(None, dtype=pl.Int64).alias("rank_l1"),
                pl.lit(None, dtype=pl.Float64).alias("z_l1"),
            ]
        )
    x = feature_matrix(survivors, L1_FEATURES)
    z = model.z(x, u, mu=config.priority.mu)
    ranked = survivors.with_columns(pl.Series("z_l1", z)).sort("z_l1", descending=True)
    ranked = ranked.with_row_index("rank_l1", offset=1)
    return df.join(ranked.select(["doc_id", "rank_l1", "z_l1"]), on="doc_id", how="left")


def _rank_l2(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    mk = config.ranking.m * config.ranking.k
    candidate = df.filter(pl.col("rank_l1").is_not_null()).sort("rank_l1").head(mk)
    base = df.with_columns(
        [
            pl.lit(None, dtype=pl.Int64).alias("rank_l2"),
            pl.lit(None, dtype=pl.Float64).alias("score_l2"),
            pl.lit(False).alias("in_top_k"),
        ]
    )
    if candidate.is_empty():
        return base
    if config.ranking.l2_mode == L2Mode.disabled:
        top = candidate.head(config.ranking.k).with_columns(pl.lit(True).alias("in_top_k"))
        return base.drop("in_top_k").join(top.select(["doc_id", "in_top_k"]), on="doc_id", how="left").with_columns(
            pl.col("in_top_k").fill_null(False)
        )

    unit_u = config.priority.as_unit_vector()
    candidate = add_interaction_features(candidate, unit_u)
    if config.ranking.l2_mode == L2Mode.prototype:
        scores = prototype_l2_score(candidate, config.prototype)
    else:
        model = _load_l2_model(config)
        scores = model.predict(l2_feature_matrix(candidate))
    ranked = (
        candidate.with_columns(pl.Series("score_l2", scores))
        .sort("score_l2", descending=True)
        .with_row_index("rank_l2", offset=1)
        .with_columns((pl.col("rank_l2") <= config.ranking.k).alias("in_top_k"))
    )
    return (
        base.drop(["rank_l2", "score_l2", "in_top_k"])
        .join(ranked.select(["doc_id", "rank_l2", "score_l2", "in_top_k"]), on="doc_id", how="left")
        .with_columns(pl.col("in_top_k").fill_null(False))
    )


def _finalize_output(df: pl.DataFrame, include_filtered_rows: bool) -> pl.DataFrame:
    output = df
    if not include_filtered_rows:
        output = output.filter(pl.col("passed_hard_filter"))
    missing = [column for column in OUTPUT_COLUMNS if column not in output.columns]
    for column in missing:
        output = output.with_columns(pl.lit(None).alias(column))
    return output.select(OUTPUT_COLUMNS).sort(
        ["in_top_k", "rank_l2", "rank_l1"],
        descending=[True, False, False],
        nulls_last=True,
    )


def run_pipeline(
    config: PipelineConfig,
    propella: pl.DataFrame | None = None,
    finepdf: pl.DataFrame | None = None,
) -> PipelineResult:
    raw, join_report = prepare_joined_frame(config, propella=propella, finepdf=finepdf)
    featured, feature_report = prepare_features(raw, config)
    l1_ranked = _rank_l1(featured, config)
    l2_ranked = _rank_l2(l1_ranked, config)
    output = _finalize_output(l2_ranked, include_filtered_rows=config.ranking.include_filtered_rows)
    report = {
        "join": join_report.to_dict(),
        "rows": {
            "total": output.height,
            "passed_hard_filter": int(output["passed_hard_filter"].sum()),
            "top_k": int(output["in_top_k"].sum()),
        },
        "modes": {
            "l1": "prototype_heuristic" if not config.l1_model_path else "trained_pairwise_linear",
            "l2": config.ranking.l2_mode.value,
        },
        **feature_report,
    }
    return PipelineResult(output=output, report=report)


def write_output(result: PipelineResult, config: PipelineConfig) -> None:
    path = config.ranking.output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if config.ranking.output_format == OutputFormat.parquet:
        result.output.write_parquet(path)
    else:
        result.output.write_ndjson(path)
    write_json(path.with_suffix(path.suffix + ".report.json"), result.report)


def preview_json(result: PipelineResult, k: int) -> str:
    rows = result.output.filter(pl.col("in_top_k")).head(k).to_dicts()
    return json.dumps(rows, indent=2, default=str)
