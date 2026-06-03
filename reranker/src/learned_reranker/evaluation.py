from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from learned_reranker.artifacts import read_pickle, write_json
from learned_reranker.config import PipelineConfig
from learned_reranker.data import canonical_doc_id
from learned_reranker.features import feature_matrix
from learned_reranker.l1 import evaluate_l1, recall_at_percentile
from learned_reranker.l2 import add_interaction_features, hash_split, l2_feature_matrix
from learned_reranker.pipeline import _rank_l1, prepare_features, prepare_joined_frame
from learned_reranker.schema import L1_FEATURES
from learned_reranker.training import _attach_labels, load_labels


def spearman(pred: np.ndarray, gold: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    pred_ranks = np.argsort(np.argsort(pred))
    gold_ranks = np.argsort(np.argsort(gold))
    return float(np.corrcoef(pred_ranks, gold_ranks)[0, 1])


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    if len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    rel = relevance[order]
    discounts = np.log2(np.arange(2, len(rel) + 2))
    dcg = float(np.sum((2**rel - 1) / discounts))
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.sum((2**ideal - 1) / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def relevance_bins_from_gold(gold: np.ndarray, bins: int = 8) -> np.ndarray:
    clipped = np.clip(gold, 0.0, 1.0)
    return np.minimum(np.floor(clipped * bins), bins - 1).astype(int)


def doc_hash_key(doc_id: str) -> int:
    digest = hashlib.sha256(canonical_doc_id(doc_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def scores_from_lexicographic_sort(
    df: pl.DataFrame,
    sort_columns: list[str],
    descending: list[bool],
) -> np.ndarray:
    n = df.height
    scores = np.zeros(n, dtype=float)
    order = (
        df.with_row_index("_row_nr")
        .sort(sort_columns, descending=descending, nulls_last=True)
        .select("_row_nr")
    )
    for rank, row_nr in enumerate(order["_row_nr"].to_list()):
        scores[int(row_nr)] = float(n - rank)
    return scores


def baseline_random_scores(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n)


def baseline_feature_mean_scores(df: pl.DataFrame) -> np.ndarray:
    return feature_matrix(df, L1_FEATURES).mean(axis=1)


def baseline_edu_quality_safety_hash_scores(df: pl.DataFrame) -> np.ndarray:
    working = df.with_columns(
        pl.col("doc_id")
        .map_elements(doc_hash_key, return_dtype=pl.UInt64)
        .alias("_doc_hash")
    )
    return scores_from_lexicographic_sort(
        working,
        ["educational_value", "content_quality", "safety", "_doc_hash"],
        descending=[True, True, True, True],
    )


def baseline_quality_edu_safety_hash_scores(df: pl.DataFrame) -> np.ndarray:
    working = df.with_columns(
        pl.col("doc_id")
        .map_elements(doc_hash_key, return_dtype=pl.UInt64)
        .alias("_doc_hash")
    )
    return scores_from_lexicographic_sort(
        working,
        ["content_quality", "educational_value", "safety", "_doc_hash"],
        descending=[True, True, True, True],
    )


def dedupe_by_doc_id(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("doc_id")
            .map_elements(canonical_doc_id, return_dtype=pl.Utf8)
            .alias("_canon_doc_id")
        )
        .unique(subset=["_canon_doc_id"], keep="first")
        .drop("_canon_doc_id")
    )


def with_data_path(config: PipelineConfig, data_path: Path) -> PipelineConfig:
    dataset = config.dataset.model_copy(update={"propella_dataset": str(data_path)})
    return config.model_copy(update={"dataset": dataset})


def prepare_labeled_frame(config: PipelineConfig, labels_path: Path) -> pl.DataFrame:
    labels = load_labels(labels_path)
    raw, _ = prepare_joined_frame(config, target_ids=set(labels["doc_id"].to_list()))
    featured, _ = prepare_features(raw, config)
    labeled = _attach_labels(featured.filter(pl.col("passed_hard_filter")), labels)
    return dedupe_by_doc_id(labeled)


def load_l1_model_weights(config: PipelineConfig) -> dict[str, float] | None:
    if not config.l1_model_path or not config.l1_model_path.exists():
        return None
    model = read_pickle(config.l1_model_path)
    weights = dict(model.weights)
    weights["intercept"] = float(model.intercept)
    return weights


def metrics_for_scores(
    scores: np.ndarray,
    gold: np.ndarray,
    percentiles: list[int],
    ndcg_ks: list[int],
    relevance_bins: int = 8,
) -> dict[str, float]:
    rel = relevance_bins_from_gold(gold, bins=relevance_bins)
    result: dict[str, float] = {"spearman": spearman(scores, gold)}
    for percentile in percentiles:
        result[f"recall_at_{percentile}pct"] = recall_at_percentile(scores, gold, percentile)
    for k in ndcg_ks:
        if k <= len(gold):
            result[f"ndcg_at_{k}"] = ndcg_at_k(scores, rel, k)
    return result


def _test_frame_with_l1_scores(
    config: PipelineConfig,
    labels_path: Path,
    test_df: pl.DataFrame,
) -> pl.DataFrame:
    labels = load_labels(labels_path)
    raw, _ = prepare_joined_frame(config, target_ids=set(labels["doc_id"].to_list()))
    featured, _ = prepare_features(raw, config)
    l1_ranked = _rank_l1(featured, config)
    ranked_labeled = dedupe_by_doc_id(
        _attach_labels(l1_ranked.filter(pl.col("passed_hard_filter")), labels)
    )
    test_canon = {
        canonical_doc_id(doc_id) for doc_id in test_df["doc_id"].to_list()
    }
    ranked_test = ranked_labeled.filter(
        pl.col("doc_id")
        .map_elements(canonical_doc_id, return_dtype=pl.Utf8)
        .is_in(list(test_canon))
    )
    return test_df.join(
        ranked_test.select(["doc_id", "z_l1"]),
        on="doc_id",
        how="left",
    )


@dataclass(frozen=True)
class MethodResult:
    key: str
    label: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class EvaluationReport:
    config_path: Path
    labels_path: Path
    generated_at: str
    n_rows_in_labels_file: int
    n_labeled_passed_filter: int
    n_test: int
    gold_mean: float
    gold_std: float
    methods: tuple[MethodResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "labels_path": str(self.labels_path),
            "generated_at": self.generated_at,
            "n_rows_in_labels_file": self.n_rows_in_labels_file,
            "n_labeled_passed_filter": self.n_labeled_passed_filter,
            "n_test": self.n_test,
            "gold_mean": self.gold_mean,
            "gold_std": self.gold_std,
            "methods": [
                {"key": method.key, "label": method.label, "metrics": method.metrics}
                for method in self.methods
            ],
        }


def evaluate_methods_on_frame(
    config: PipelineConfig,
    labels_path: Path,
    test_df: pl.DataFrame,
    *,
    seed: int = 13,
) -> tuple[MethodResult, ...]:
    gold = test_df["gold_score"].to_numpy()
    percentiles = config.l1_train.eval_percentiles
    ndcg_ks = config.l2_train.eval_k

    methods: list[MethodResult] = [
        MethodResult(
            key="random_shuffle",
            label="Random shuffle",
            metrics=metrics_for_scores(
                baseline_random_scores(len(gold), seed=seed),
                gold,
                percentiles,
                ndcg_ks,
            ),
        ),
        MethodResult(
            key="feature_mean",
            label="Descending mean of L1 input features",
            metrics=metrics_for_scores(
                baseline_feature_mean_scores(test_df),
                gold,
                percentiles,
                ndcg_ks,
            ),
        ),
        MethodResult(
            key="edu_quality_safety_hash",
            label="educational_value → content_quality → safety → doc_hash",
            metrics=metrics_for_scores(
                baseline_edu_quality_safety_hash_scores(test_df),
                gold,
                percentiles,
                ndcg_ks,
            ),
        ),
        MethodResult(
            key="quality_edu_safety_hash",
            label="content_quality → educational_value → safety → doc_hash",
            metrics=metrics_for_scores(
                baseline_quality_edu_safety_hash_scores(test_df),
                gold,
                percentiles,
                ndcg_ks,
            ),
        ),
    ]

    test_ranked = _test_frame_with_l1_scores(config, labels_path, test_df)
    if config.l1_model_path and config.l1_model_path.exists() and not test_ranked.is_empty():
        z = test_ranked["z_l1"].to_numpy()
        y = test_ranked["gold_score"].to_numpy()
        l1_model = read_pickle(config.l1_model_path)
        l1_metrics = evaluate_l1(l1_model, feature_matrix(test_ranked, L1_FEATURES), y, percentiles)
        l1_metrics["spearman"] = spearman(z, y)
        rel = relevance_bins_from_gold(y)
        for k in ndcg_ks:
            if k <= len(y):
                l1_metrics[f"ndcg_at_{k}"] = ndcg_at_k(z, rel, k)
        methods.append(
            MethodResult(
                key="l1_trained",
                label="Trained L1 (pairwise linear)",
                metrics=l1_metrics,
            )
        )

    if config.l2_model_path and config.l2_model_path.exists() and not test_ranked.is_empty():
        y = test_ranked["gold_score"].to_numpy()
        unit_u = config.priority.as_unit_vector()
        l2_df = add_interaction_features(test_ranked, unit_u)
        l2_scores = read_pickle(config.l2_model_path).predict(l2_feature_matrix(l2_df))
        l2_metrics = metrics_for_scores(l2_scores, y, percentiles, ndcg_ks)
        methods.append(
            MethodResult(
                key="l2_trained",
                label="Trained L2 (LightGBM LambdaRank)",
                metrics=l2_metrics,
            )
        )
    return tuple(methods)


def _build_evaluation_report(
    config_path: Path,
    labels_path: Path,
    labels_file_rows: int,
    labeled_count: int,
    test_df: pl.DataFrame,
    methods: tuple[MethodResult, ...],
) -> EvaluationReport:
    gold = test_df["gold_score"].to_numpy()
    return EvaluationReport(
        config_path=config_path,
        labels_path=labels_path,
        generated_at=datetime.now(timezone.utc).isoformat(),
        n_rows_in_labels_file=labels_file_rows,
        n_labeled_passed_filter=labeled_count,
        n_test=test_df.height,
        gold_mean=float(np.mean(gold)),
        gold_std=float(np.std(gold)),
        methods=methods,
    )


def run_evaluation(
    config: PipelineConfig,
    labels_path: Path,
    *,
    seed: int = 13,
    config_path: Path | None = None,
) -> EvaluationReport:
    labels = load_labels(labels_path)
    labeled = prepare_labeled_frame(config, labels_path)
    if labeled.is_empty():
        raise ValueError("No labeled documents remain after hard filter and deduplication")

    split_masks = hash_split(labeled["doc_id"].to_list())
    test_df = labeled.filter(pl.Series(split_masks.test))
    if test_df.is_empty():
        raise ValueError("No documents in the hash holdout test split (10%)")

    methods = evaluate_methods_on_frame(config, labels_path, test_df, seed=seed)
    return _build_evaluation_report(
        config_path or Path("."),
        labels_path,
        labels.height,
        labeled.height,
        test_df,
        methods,
    )


def run_evaluation_external(
    config: PipelineConfig,
    labels_path: Path,
    data_path: Path,
    *,
    seed: int = 13,
    config_path: Path | None = None,
) -> EvaluationReport:
    eval_config = with_data_path(config, data_path)
    labels = load_labels(labels_path)
    labeled = prepare_labeled_frame(eval_config, labels_path)
    if labeled.is_empty():
        raise ValueError("No labeled documents remain after hard filter and deduplication")
    methods = evaluate_methods_on_frame(eval_config, labels_path, labeled, seed=seed)
    return _build_evaluation_report(
        config_path or Path("."),
        labels_path,
        labels.height,
        labeled.height,
        labeled,
        methods,
    )


def _format_metric(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.3f}"


def _metric_keys(methods: tuple[MethodResult, ...]) -> list[str]:
    keys: list[str] = ["spearman"]
    for method in methods:
        for key in method.metrics:
            if key not in keys:
                keys.append(key)
    return keys


def _render_results_table(methods: tuple[MethodResult, ...]) -> list[str]:
    metric_keys = _metric_keys(methods)
    header = "| Method | " + " | ".join(metric_keys) + " |"
    separator = "| --- | " + " | ".join(["---:"] * len(metric_keys)) + " |"
    lines = [header, separator]
    for method in methods:
        cells = [_format_metric(method.metrics.get(key, float("nan"))) for key in metric_keys]
        lines.append(f"| {method.label} | " + " | ".join(cells) + " |")
    return lines


def _render_dataset_section(
    report: EvaluationReport,
    *,
    split_description: str,
) -> list[str]:
    return [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Labels file | `{report.labels_path}` |",
        f"| Rows in labels file | {report.n_rows_in_labels_file} |",
        f"| Labeled + passed hard filter (deduped) | {report.n_labeled_passed_filter} |",
        f"| Evaluation documents | {report.n_test} |",
        f"| Gold score mean | {report.gold_mean:.3f} |",
        f"| Gold score std | {report.gold_std:.3f} |",
        "",
        f"Split: {split_description}. Hard filter uses `full_doc_lid == expected_lid` from config.",
        "",
    ]


def _render_l1_weights_section(weights: dict[str, float]) -> list[str]:
    feature_weights = [(name, weights[name]) for name in L1_FEATURES if name in weights]
    feature_weights.sort(key=lambda item: abs(item[1]), reverse=True)
    lines = [
        "## Trained L1 weights",
        "",
        "Linear pairwise ranker feature weights (higher absolute value = stronger influence on `z_l1`).",
        "",
        "| Feature | Weight |",
        "| --- | ---: |",
    ]
    for name, value in feature_weights:
        lines.append(f"| `{name}` | {value:.6f} |")
    intercept = weights.get("intercept")
    if intercept is not None:
        lines.extend(["", f"**Intercept:** {intercept:.6f}", ""])
    return lines


@dataclass(frozen=True)
class FullEvaluationReport:
    training_labels_path: Path
    holdout: EvaluationReport
    external_evals: tuple[tuple[str, EvaluationReport], ...] = ()
    l1_weights: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "training_labels_path": str(self.training_labels_path),
            "holdout": self.holdout.to_dict(),
            "l1_weights": self.l1_weights,
        }
        if self.external_evals:
            payload["external_evals"] = {
                name: report.to_dict() for name, report in self.external_evals
            }
        return payload


def render_markdown_report(
    full_report: FullEvaluationReport,
    *,
    config_path: Path,
) -> str:
    holdout = full_report.holdout
    lines = [
        "# Evaluation Report",
        "",
        f"**Generated:** {holdout.generated_at}",
        f"**Config:** `{config_path}`",
        f"**Training data:** `{full_report.training_labels_path}`",
        "",
    ]
    if full_report.l1_weights:
        lines.extend(_render_l1_weights_section(full_report.l1_weights))
        lines.append("")

    lines.extend(
        [
            "## Holdout test (training data)",
            "",
            *_render_dataset_section(
                holdout,
                split_description="hash holdout (10% test) from training labels",
            ),
            *_render_results_table(holdout.methods),
            "",
        ]
    )

    for section_name, section_report in full_report.external_evals:
        lines.extend(
            [
                f"## {section_name}",
                "",
                *_render_dataset_section(
                    section_report,
                    split_description="all labeled documents (external evaluation)",
                ),
                *_render_results_table(section_report.methods),
                "",
            ]
        )

    lines.extend(
        [
            "## Methods",
            "",
            "| Key | Description |",
            "| --- | --- |",
            "| `random_shuffle` | Uniform random scores (fixed seed) |",
            "| `feature_mean` | Mean of normalized L1 input features (`*_z`) |",
            "| `edu_quality_safety_hash` | Lexicographic: educational_value, content_quality, safety, doc_hash |",
            "| `quality_edu_safety_hash` | Lexicographic: content_quality, educational_value, safety, doc_hash |",
            "| `l1_trained` | Trained pairwise linear L1 model (`z_l1`) |",
            "| `l2_trained` | Trained LightGBM LambdaRank L2 model |",
            "",
            "Recall@p%: fraction of top-p% gold documents captured in the top-p% of predicted scores. "
            "NDCG uses 8-bin relevance derived from gold scores in `[0, 1]`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_report(
    full_report: FullEvaluationReport,
    *,
    config_path: Path,
    markdown_path: Path,
    json_path: Path | None = None,
) -> FullEvaluationReport:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_markdown_report(full_report, config_path=config_path),
        encoding="utf-8",
    )
    if json_path is not None:
        write_json(json_path, full_report.to_dict())
    return full_report


def run_full_evaluation(
    config: PipelineConfig,
    training_labels_path: Path,
    *,
    config_path: Path,
    seed: int = 13,
    external_evals: tuple[tuple[str, Path, Path], ...] = (),
) -> FullEvaluationReport:
    """Holdout eval on training labels plus optional external (name, labels, data) sets."""
    holdout = run_evaluation(
        config,
        training_labels_path,
        seed=seed,
        config_path=config_path,
    )
    externals: list[tuple[str, EvaluationReport]] = []
    for name, labels_path, data_path in external_evals:
        externals.append(
            (
                name,
                run_evaluation_external(
                    config,
                    labels_path,
                    data_path,
                    seed=seed,
                    config_path=config_path,
                ),
            )
        )
    return FullEvaluationReport(
        training_labels_path=training_labels_path,
        holdout=holdout,
        external_evals=tuple(externals),
        l1_weights=load_l1_model_weights(config),
    )
