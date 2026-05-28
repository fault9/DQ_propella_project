from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from learned_reranker.config import L1TrainConfig, PrototypeConfig
from learned_reranker.schema import L1_FEATURES


@dataclass
class L1Model:
    weights: dict[str, float]
    intercept: float = 0.0
    w_max: float = 1.0
    model_type: str = "trained_pairwise_linear"

    def z(self, x: np.ndarray, u: np.ndarray, mu: float | None = None) -> np.ndarray:
        weights = np.array([self.weights[name] for name in L1_FEATURES], dtype=float)
        personal_mu = self.w_max if mu is None else mu
        return x @ weights + personal_mu * (x @ u) + self.intercept


def prototype_l1_model(config: PrototypeConfig) -> L1Model:
    # PROTOTYPE: fixed transparent weights until universal gold-quality labels are available.
    weights = {name: float(config.l1_weights[name]) for name in L1_FEATURES}
    return L1Model(
        weights=weights,
        intercept=0.0,
        w_max=max(abs(value) for value in weights.values()),
        model_type="prototype_heuristic",
    )


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35, 35)))


def _pairs(y: np.ndarray, max_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    candidate_pairs = [(i, j) for i, j in combinations(range(len(y)), 2) if y[i] != y[j]]
    if len(candidate_pairs) > max_pairs:
        idx = rng.choice(len(candidate_pairs), size=max_pairs, replace=False)
        return [candidate_pairs[int(i)] for i in idx]
    return candidate_pairs


def fit_pairwise_linear_ranker(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    gamma: float,
    config: L1TrainConfig,
    seed: int = 13,
) -> L1Model:
    if x.shape[1] != len(L1_FEATURES):
        raise ValueError(f"Expected {len(L1_FEATURES)} L1 features, got {x.shape[1]}")
    rng = np.random.default_rng(seed)
    pairs = _pairs(y, config.max_pairs, rng)
    weights = np.zeros(x.shape[1], dtype=float)
    intercept = 0.0
    lr = config.learning_rate
    if pairs:
        pair_array = np.array(pairs, dtype=int)
        left = pair_array[:, 0]
        right = pair_array[:, 1]
        swap = y[right] > y[left]
        winners = np.where(swap, right, left)
        losers = np.where(swap, left, right)
        diff_x_all = x[winners] - x[losers]
        pair_weights = 1.0 + gamma * np.maximum(y[winners] - y[losers], 0.0)
    else:
        diff_x_all = np.empty((0, x.shape[1]), dtype=float)
        pair_weights = np.empty(0, dtype=float)
    for _ in range(config.epochs):
        grad_w = alpha * weights
        margins = diff_x_all @ weights
        # d -log(sigmoid(margin)) / d margin = sigmoid(margin) - 1
        grad_margins = pair_weights * (sigmoid(margins) - 1.0)
        grad_w += grad_margins @ diff_x_all
        grad_b = float(grad_margins.sum())
        scale = max(len(pairs), 1)
        weights -= lr * grad_w / scale
        intercept -= lr * grad_b / scale
    return L1Model(
        weights={name: float(weights[idx]) for idx, name in enumerate(L1_FEATURES)},
        intercept=float(intercept),
        w_max=float(max(np.max(np.abs(weights)), 1e-6)),
    )


def recall_at_percentile(z: np.ndarray, y: np.ndarray, percentile: int) -> float:
    if len(y) == 0:
        return 0.0
    cutoff = max(1, int(np.ceil(len(y) * percentile / 100.0)))
    top_pred = set(np.argsort(-z)[:cutoff].tolist())
    top_gold = set(np.argsort(-y)[:cutoff].tolist())
    return len(top_pred & top_gold) / max(len(top_gold), 1)


def evaluate_l1(model: L1Model, x: np.ndarray, y: np.ndarray, percentiles: list[int]) -> dict[str, float]:
    zeros = np.zeros(x.shape[1], dtype=float)
    z = model.z(x, zeros, mu=0.0)
    return {f"recall_at_{p}pct": recall_at_percentile(z, y, p) for p in percentiles}
