import polars as pl

from learned_reranker.config import HardFilter, PriorityConfig
from learned_reranker.features import (
    add_l2_dedup_features,
    apply_hard_filter,
    apply_percentile_normalization,
    convert_raw_features,
    fit_normalization_stats,
)
from learned_reranker.schema import L1_FEATURES


def raw_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["a", "b"],
            "content_quality": ["very_high", "low"],
            "information_density": ["dense", "sparse"],
            "educational_value": ["high", "low"],
            "safety": ["safe", "unsafe"],
            "pii_presence": ["no_pii", "contains_pii"],
            "content_length": ["substantial", "brief"],
            "content_type": ["report", "report"],
            "full_doc_lid": ["swe_Latn", "eng_Latn"],
            "full_doc_lid_score": [0.99, 0.88],
            "minhash_cluster_size": [1, 10],
            "duplicate_count": [0, 4],
        }
    )


def test_feature_conversion_inverts_pii_presence() -> None:
    converted = convert_raw_features(raw_frame())
    assert converted["pii_absence"].to_list() == [1, 0]
    assert converted["length"].to_list() == [3, 1]
    assert converted["language_confidence"].to_list() == [0.99, 0.88]


def test_default_hard_filter_only_expected_language() -> None:
    converted = convert_raw_features(raw_frame())
    filtered = apply_hard_filter(converted, HardFilter(expected_lid="swe_Latn"))
    assert filtered["passed_hard_filter"].to_list() == [True, False]


def test_priority_budget_must_sum_to_100() -> None:
    config = PriorityConfig()
    assert round(sum(config.credits.values()), 6) == 100.0


def test_normalization_and_dedup_internal_features() -> None:
    converted = convert_raw_features(raw_frame())
    stats = fit_normalization_stats(converted, L1_FEATURES)
    normalized = apply_percentile_normalization(converted, stats, L1_FEATURES)
    deduped = add_l2_dedup_features(normalized)
    assert "inverted_cluster_size" in deduped.columns
    assert "inverted_duplicate_count" in deduped.columns
    assert deduped["inverted_cluster_size"].min() >= 0.0
    assert deduped["inverted_cluster_size"].max() <= 1.0
