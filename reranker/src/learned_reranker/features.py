from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import polars as pl

from learned_reranker.config import HardFilter
from learned_reranker.schema import L1_FEATURES


CONTENT_QUALITY = {
    "unacceptable": 0,
    "very_low": 0,
    "poor": 1,
    "low": 1,
    "adequate": 2,
    "medium": 2,
    "good": 3,
    "high": 3,
    "excellent": 4,
    "very_high": 4,
}

INFORMATION_DENSITY = {
    "empty": 0,
    "very_sparse": 0,
    "thin": 1,
    "sparse": 1,
    "adequate": 2,
    "moderate": 2,
    "dense": 3,
    "very_dense": 4,
}

EDUCATIONAL_VALUE = {
    "none": 0,
    "very_low": 0,
    "minimal": 1,
    "low": 1,
    "basic": 2,
    "medium": 2,
    "moderate": 3,
    "high": 3,
    "very_high": 4,
}

SAFETY = {
    "very_unsafe": 0,
    "unsafe": 1,
    "mild_concerns": 2,
    "mixed": 2,
    "safe": 3,
    "very_safe": 4,
}

CONTENT_LENGTH = {
    "minimal": 0,
    "brief": 1,
    "moderate": 2,
    "substantial": 3,
}

PII_ABSENCE = {
    # Propella stores pii_presence; the model uses its inverse so higher remains better.
    "contains_pii": False,
    "has_pii": False,
    "pii_present": False,
    "no_pii": True,
    "does_not_contain_pii": True,
    "pii_absent": True,
}


ENUM_MAPPINGS = {
    "content_quality": CONTENT_QUALITY,
    "information_density": INFORMATION_DENSITY,
    "educational_value": EDUCATIONAL_VALUE,
    "safety": SAFETY,
    "content_length": CONTENT_LENGTH,
}


@dataclass(frozen=True)
class NormalizationStats:
    p1: dict[str, float]
    p99: dict[str, float]
    mean: dict[str, float]
    std: dict[str, float]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "p1": self.p1,
            "p99": self.p99,
            "mean": self.mean,
            "std": self.std,
        }


def _map_enum_expr(column: str, mapping: dict[str, int]) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .replace_strict(mapping, default=None)
        .cast(pl.Int64)
    )


def _pii_absence_expr() -> pl.Expr:
    normalized = pl.col("pii_presence").cast(pl.Utf8).str.to_lowercase()
    return (
        pl.when(normalized.is_in(["contains_pii", "has_pii", "pii_present", "true", "1"]))
        .then(False)
        .when(normalized.is_in(["no_pii", "does_not_contain_pii", "pii_absent", "false", "0"]))
        .then(True)
        .otherwise(None)
        .alias("pii_absence")
    )


def convert_raw_features(df: pl.DataFrame) -> pl.DataFrame:
    """Convert Propella/FinePDF raw columns into model-facing numeric features."""
    safety_column = "safety" if "safety" in df.columns else "content_safety"
    converted = df.with_columns(
        [
            _map_enum_expr("content_quality", CONTENT_QUALITY).alias("content_quality"),
            _map_enum_expr("information_density", INFORMATION_DENSITY).alias(
                "information_density"
            ),
            _map_enum_expr("educational_value", EDUCATIONAL_VALUE).alias("educational_value"),
            _map_enum_expr(safety_column, SAFETY).alias("safety"),
            _map_enum_expr("content_length", CONTENT_LENGTH).alias("length"),
            _pii_absence_expr(),
            pl.col("full_doc_lid_score").cast(pl.Float64).alias("language_confidence"),
            pl.col("minhash_cluster_size").fill_null(0).cast(pl.Float64),
            pl.col("duplicate_count").fill_null(0).cast(pl.Float64),
        ]
    )
    return converted.with_columns(
        [
            pl.col("pii_absence").cast(pl.Int64),
            pl.col("id").alias("doc_id"),
        ]
    )


def apply_hard_filter(df: pl.DataFrame, hard_filter: HardFilter) -> pl.DataFrame:
    predicates: list[pl.Expr] = [pl.col("full_doc_lid") == hard_filter.expected_lid]
    threshold_map = {
        "content_quality": hard_filter.min_content_quality,
        "information_density": hard_filter.min_information_density,
        "educational_value": hard_filter.min_educational_value,
        "safety": hard_filter.min_safety,
        "length": hard_filter.min_length,
        "language_confidence": hard_filter.min_language_confidence,
    }
    for column, value in threshold_map.items():
        if value is not None:
            predicates.append(pl.col(column) >= value)
    if hard_filter.require_pii_absence is not None:
        predicates.append(pl.col("pii_absence") == int(hard_filter.require_pii_absence))

    predicate = predicates[0]
    for next_predicate in predicates[1:]:
        predicate = predicate & next_predicate
    return df.with_columns(predicate.alias("passed_hard_filter"))


def fit_normalization_stats(df: pl.DataFrame, columns: Iterable[str]) -> NormalizationStats:
    p1: dict[str, float] = {}
    p99: dict[str, float] = {}
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for column in columns:
        series = df[column].drop_nulls().cast(pl.Float64)
        if series.is_empty():
            low = high = avg = 0.0
            sigma = 1.0
        else:
            low = float(series.quantile(0.01, interpolation="nearest"))
            high = float(series.quantile(0.99, interpolation="nearest"))
            avg = float(series.mean())
            sigma = float(series.std() or 1.0)
        if high <= low:
            high = low + 1.0
        p1[column] = low
        p99[column] = high
        mean[column] = avg
        std[column] = sigma
    return NormalizationStats(p1=p1, p99=p99, mean=mean, std=std)


def apply_percentile_normalization(
    df: pl.DataFrame,
    stats: NormalizationStats,
    columns: Iterable[str],
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for column in columns:
        low = stats.p1[column]
        high = stats.p99[column]
        expressions.append(
            (((pl.col(column).cast(pl.Float64) - low) / (high - low)).clip(0.0, 1.0)).alias(
                f"{column}_z"
            )
        )
    return df.with_columns(expressions)


def add_l2_dedup_features(df: pl.DataFrame, stats: NormalizationStats | None = None) -> pl.DataFrame:
    working = df.with_columns(
        [
            (pl.col("minhash_cluster_size").cast(pl.Float64) + 1.0).log().alias(
                "log_cluster_size"
            ),
            (pl.col("duplicate_count").cast(pl.Float64) + 1.0).log().alias(
                "log_duplicate_count"
            ),
        ]
    )
    local_stats = stats or fit_normalization_stats(
        working, ["log_cluster_size", "log_duplicate_count"]
    )
    normalized = apply_percentile_normalization(
        working, local_stats, ["log_cluster_size", "log_duplicate_count"]
    )
    return normalized.with_columns(
        [
            (1.0 - pl.col("log_cluster_size_z")).alias("inverted_cluster_size"),
            (1.0 - pl.col("log_duplicate_count_z")).alias("inverted_duplicate_count"),
        ]
    )


def feature_matrix(df: pl.DataFrame, features: Iterable[str]) -> np.ndarray:
    columns = []
    for feature in features:
        column = f"{feature}_z" if feature in L1_FEATURES else feature
        columns.append(column)
    return df.select(columns).fill_null(0.0).to_numpy().astype(float)


def priority_vector(unit_credits: dict[str, float]) -> np.ndarray:
    return np.array([unit_credits[name] for name in L1_FEATURES], dtype=float)
