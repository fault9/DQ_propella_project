import numpy as np
import polars as pl

from learned_reranker.evaluation import (
    baseline_edu_quality_safety_hash_scores,
    baseline_feature_mean_scores,
    baseline_quality_edu_safety_hash_scores,
    baseline_random_scores,
    metrics_for_scores,
    scores_from_lexicographic_sort,
    spearman,
)


def test_lexicographic_sort_orders_by_primary_key() -> None:
    df = pl.DataFrame(
        {
            "content_quality": [1, 3, 3],
            "educational_value": [3, 1, 2],
            "safety": [1, 1, 1],
            "doc_id": ["a", "b", "c"],
        }
    )
    scores = scores_from_lexicographic_sort(
        df,
        ["content_quality", "educational_value", "safety", "doc_id"],
        descending=[True, True, True, True],
    )
    assert scores[2] > scores[1] > scores[0]


def test_edu_before_quality_baseline_differs_from_quality_first() -> None:
    df = pl.DataFrame(
        {
            "content_quality": [2, 4, 2],
            "educational_value": [4, 2, 2],
            "safety": [1, 1, 1],
            "doc_id": ["a", "b", "c"],
        }
    )
    edu_first = baseline_edu_quality_safety_hash_scores(df)
    quality_first = baseline_quality_edu_safety_hash_scores(df)
    assert not np.array_equal(edu_first, quality_first)
    assert edu_first[0] > edu_first[1]
    assert quality_first[1] > quality_first[0]


def test_random_baseline_is_reproducible() -> None:
    a = baseline_random_scores(20, seed=7)
    b = baseline_random_scores(20, seed=7)
    c = baseline_random_scores(20, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_feature_mean_prefers_higher_normalized_features() -> None:
    df = pl.DataFrame(
        {
            f"{name}_z": [0.1, 0.9]
            for name in [
                "content_quality",
                "information_density",
                "educational_value",
                "safety",
                "pii_absence",
                "length",
                "language_confidence",
            ]
        }
        | {"doc_id": ["low", "high"]}
    )
    scores = baseline_feature_mean_scores(df)
    assert scores[1] > scores[0]


def test_spearman_is_perfect_for_identical_ranking() -> None:
    values = np.array([0.1, 0.4, 0.9])
    assert spearman(values, values) == 1.0


def test_metrics_for_scores_includes_recall_and_ndcg() -> None:
    scores = np.array([3.0, 2.0, 1.0])
    gold = np.array([0.9, 0.5, 0.1])
    metrics = metrics_for_scores(scores, gold, percentiles=[50], ndcg_ks=[3])
    assert "spearman" in metrics
    assert "recall_at_50pct" in metrics
    assert "ndcg_at_3" in metrics
