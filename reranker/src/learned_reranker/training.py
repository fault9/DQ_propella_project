from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from learned_reranker.artifacts import write_json, write_pickle
from learned_reranker.config import PipelineConfig
from learned_reranker.data import canonical_doc_id
from learned_reranker.features import feature_matrix
from learned_reranker.l1 import evaluate_l1, fit_pairwise_linear_ranker
from learned_reranker.l2 import hash_split, relevance_bins, train_lambdarank
from learned_reranker.pipeline import prepare_features, prepare_joined_frame
from learned_reranker.schema import L1_FEATURES


def load_labels(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        labels = pl.read_parquet(path)
    else:
        labels = pl.read_csv(path)
    id_column = "id" if "id" in labels.columns else "doc_id" if "doc_id" in labels.columns else None
    score_column = (
        "gold_score"
        if "gold_score" in labels.columns
        else "score"
        if "score" in labels.columns
        else "quality_score"
        if "quality_score" in labels.columns
        else None
    )
    if id_column is None or score_column is None:
        raise ValueError(
            "Labels must contain id/doc_id and gold_score/score/quality_score columns"
        )
    return labels.select(
        [
            pl.col(id_column).cast(pl.Utf8).alias("doc_id"),
            pl.col(score_column).cast(pl.Float64).clip(0.0, 1.0).alias("gold_score"),
        ]
    )


def _attach_labels(df: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    keyed_df = df.with_columns(
        pl.col("doc_id").map_elements(canonical_doc_id, return_dtype=pl.Utf8).alias("_join_doc_id")
    )
    keyed_labels = labels.rename({"doc_id": "label_doc_id"}).with_columns(
        pl.col("label_doc_id")
        .map_elements(canonical_doc_id, return_dtype=pl.Utf8)
        .alias("_join_doc_id")
    )
    return keyed_df.join(keyed_labels, on="_join_doc_id", how="inner").drop(
        ["_join_doc_id", "label_doc_id"]
    )


def _label_distribution(y: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(len(y)),
        "min": float(np.min(y)) if len(y) else None,
        "max": float(np.max(y)) if len(y) else None,
        "mean": float(np.mean(y)) if len(y) else None,
        "std": float(np.std(y)) if len(y) else None,
        "quartiles": np.quantile(y, [0.25, 0.5, 0.75]).tolist() if len(y) else [],
    }


def _bin_histogram(y: np.ndarray, bins: int) -> dict[str, int]:
    rel = relevance_bins(y, bins=bins)
    return {str(idx): int((rel == idx).sum()) for idx in range(bins)}


def train_l1(
    config: PipelineConfig,
    labels_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    labels = load_labels(labels_path)
    raw, join_report = prepare_joined_frame(config, target_ids=set(labels["doc_id"].to_list()))
    featured, feature_report = prepare_features(raw, config)
    train_df = _attach_labels(featured.filter(pl.col("passed_hard_filter")), labels)
    if train_df.height < 2:
        raise ValueError("Need at least two labeled documents after filtering")

    masks = hash_split(train_df["doc_id"].to_list())
    x_all = feature_matrix(train_df, L1_FEATURES)
    y_all = train_df["gold_score"].to_numpy()
    x_train = x_all[masks.train]
    y_train = y_all[masks.train]
    x_val = x_all[masks.val]
    y_val = y_all[masks.val]
    best: tuple[float, Any, dict[str, float], float, float] | None = None
    for alpha in config.l1_train.alpha_grid:
        for gamma in config.l1_train.gamma_grid:
            model = fit_pairwise_linear_ranker(
                x_train,
                y_train,
                alpha=alpha,
                gamma=gamma,
                config=config.l1_train,
            )
            metrics = evaluate_l1(model, x_val, y_val, config.l1_train.eval_percentiles)
            score = float(np.mean(list(metrics.values()))) if metrics else 0.0
            if best is None or score > best[0]:
                best = (score, model, metrics, alpha, gamma)
    assert best is not None
    _, model, metrics, alpha, gamma = best
    write_pickle(output_path, model)
    report = {
        "model_type": model.model_type,
        "best_alpha": alpha,
        "best_gamma": gamma,
        "validation": metrics,
        "join": join_report.to_dict(),
        "features": feature_report,
        "feature_means": feature_report["l1_normalization"]["mean"],
        "feature_stds": feature_report["l1_normalization"]["std"],
        "label_distribution": _label_distribution(y_all),
    }
    write_json(output_path.with_suffix(output_path.suffix + ".report.json"), report)
    return report


def train_l2(
    config: PipelineConfig,
    labels_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    from learned_reranker.l2 import add_interaction_features
    from learned_reranker.pipeline import _rank_l1, prepare_features

    labels = load_labels(labels_path)
    raw, join_report = prepare_joined_frame(config, target_ids=set(labels["doc_id"].to_list()))
    featured, feature_report = prepare_features(raw, config)
    l1_ranked = _rank_l1(featured, config)
    train_df = _attach_labels(l1_ranked.filter(pl.col("passed_hard_filter")), labels)
    if train_df.height < config.l2_train.group_size:
        raise ValueError("Need more labeled documents for L2 grouped training")
    unit_u = config.priority.as_unit_vector()
    train_df = add_interaction_features(train_df, unit_u)
    model = train_lambdarank(train_df, train_df["gold_score"].to_numpy(), config.l2_train)
    write_pickle(output_path, model)
    y = train_df["gold_score"].to_numpy()
    report = {
        "model_type": "lightgbm_lambdarank",
        "join": join_report.to_dict(),
        "features": feature_report,
        "feature_means": feature_report["l1_normalization"]["mean"],
        "feature_stds": feature_report["l1_normalization"]["std"],
        "label_distribution": _label_distribution(y),
        "relevance_bin_histogram": _bin_histogram(y, config.l2_train.relevance_bins),
        "eval_k": config.l2_train.eval_k,
        "bucket_features": ["content_type", "length"],
    }
    write_json(output_path.with_suffix(output_path.suffix + ".report.json"), report)
    return report
