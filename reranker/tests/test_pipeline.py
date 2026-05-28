import polars as pl

from learned_reranker.config import L2Mode, PipelineConfig
from learned_reranker.pipeline import preview_json, run_pipeline
from learned_reranker.schema import OUTPUT_COLUMNS


def synthetic_propella() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "content_quality": ["very_high", "high", "medium", "low"],
            "information_density": ["dense", "dense", "moderate", "sparse"],
            "educational_value": ["very_high", "high", "medium", "low"],
            "safety": ["very_safe", "safe", "safe", "unsafe"],
            "pii_presence": ["no_pii", "no_pii", "no_pii", "contains_pii"],
            "content_length": ["substantial", "moderate", "brief", "brief"],
            "content_type": ["report", "report", "manual", "manual"],
        }
    )


def synthetic_finepdf() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["a", "b", "c", "d", "unmatched"],
            "full_doc_lid": ["swe_Latn", "swe_Latn", "swe_Latn", "eng_Latn", "swe_Latn"],
            "full_doc_lid_score": [0.99, 0.97, 0.95, 0.93, 0.9],
            "minhash_cluster_size": [1, 2, 5, 1, 1],
            "duplicate_count": [0, 1, 2, 0, 0],
        }
    )


def test_pipeline_output_schema_and_preview() -> None:
    config = PipelineConfig()
    config.ranking.k = 2
    config.ranking.m = 2
    config.ranking.l2_mode = L2Mode.prototype
    result = run_pipeline(config, propella=synthetic_propella(), finepdf=synthetic_finepdf())
    assert result.output.columns == list(OUTPUT_COLUMNS)
    assert result.report["join"]["dropped_unmatched_finepdf"] == 1
    assert result.output["in_top_k"].sum() == 2
    assert "inverted_cluster_size" not in result.output.columns
    assert "minhash_cluster_size" in result.output.columns
    assert preview_json(result, 2).startswith("[")


def test_l2_disabled_uses_l1_for_top_k_without_rank_l2() -> None:
    config = PipelineConfig()
    config.ranking.k = 1
    config.ranking.m = 1
    config.ranking.l2_mode = L2Mode.disabled
    result = run_pipeline(config, propella=synthetic_propella(), finepdf=synthetic_finepdf())
    assert result.output["in_top_k"].sum() == 1
    top = result.output.filter(pl.col("in_top_k")).row(0, named=True)
    assert top["rank_l2"] is None
