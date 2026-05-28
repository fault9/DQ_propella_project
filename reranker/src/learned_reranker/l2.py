from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import polars as pl

from learned_reranker.config import L2TrainConfig, PrototypeConfig
from learned_reranker.schema import L1_FEATURES, L2_EXTRA_FEATURES


def add_interaction_features(df: pl.DataFrame, unit_credits: dict[str, float]) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col(f"{name}_z") * float(unit_credits[name])).alias(f"ux_{name}")
            for name in L1_FEATURES
        ]
    )


def l2_feature_columns() -> list[str]:
    return (
        [f"{name}_z" for name in L1_FEATURES]
        + ["z_l1"]
        + [f"ux_{name}" for name in L1_FEATURES]
        + list(L2_EXTRA_FEATURES)
    )


def l2_feature_matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(l2_feature_columns()).fill_null(0.0).to_numpy().astype(float)


def relevance_bins(y: np.ndarray, bins: int = 8) -> np.ndarray:
    clipped = np.clip(y, 0.0, 1.0)
    return np.minimum(np.floor(clipped * bins), bins - 1).astype(int)


def prototype_l2_score(df: pl.DataFrame, config: PrototypeConfig) -> np.ndarray:
    # PROTOTYPE: weak reranking signal. Trained LightGBM should replace this when labels arrive.
    z = df["z_l1"].to_numpy().astype(float)
    cluster = df["inverted_cluster_size"].to_numpy().astype(float)
    duplicate = df["inverted_duplicate_count"].to_numpy().astype(float)
    return (
        z
        + float(config.l2_extra_weights["inverted_cluster_size"]) * cluster
        + float(config.l2_extra_weights["inverted_duplicate_count"]) * duplicate
    )


@dataclass(frozen=True)
class SplitMasks:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def hash_split(ids: list[str], train_ratio: float = 0.8, val_ratio: float = 0.1) -> SplitMasks:
    buckets = np.array(
        [
            int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000
            / 10_000.0
            for doc_id in ids
        ]
    )
    return SplitMasks(
        train=buckets < train_ratio,
        val=(buckets >= train_ratio) & (buckets < train_ratio + val_ratio),
        test=buckets >= train_ratio + val_ratio,
    )


def sample_groups(df: pl.DataFrame, config: L2TrainConfig, seed: int = 13) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    groups: list[np.ndarray] = []
    bucket_cols = ["content_type", "length"]
    indexed = df.with_row_index("__row_nr")
    for _, bucket in indexed.group_by(bucket_cols, maintain_order=False):
        indices = bucket["__row_nr"].to_numpy()
        if len(indices) < 2:
            continue
        for _ in range(config.groups_per_bucket):
            sample_size = min(config.group_size, len(indices))
            groups.append(rng.choice(indices, size=sample_size, replace=False))
    return groups


def train_lambdarank(
    df: pl.DataFrame,
    labels: np.ndarray,
    config: L2TrainConfig,
):
    import lightgbm as lgb

    x = l2_feature_matrix(df)
    y = relevance_bins(labels, bins=config.relevance_bins)
    groups = sample_groups(df, config)
    if not groups:
        raise ValueError("No L2 training groups could be sampled")
    x_grouped = np.concatenate([x[group] for group in groups], axis=0)
    y_grouped = np.concatenate([y[group] for group in groups], axis=0)
    group_sizes = [len(group) for group in groups]
    train_data = lgb.Dataset(x_grouped, label=y_grouped, group=group_sizes)
    params = dict(config.lightgbm_params)
    params["eval_at"] = config.eval_k
    return lgb.train(params, train_data, num_boost_round=200)
