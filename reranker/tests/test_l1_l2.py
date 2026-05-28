import numpy as np
import polars as pl

from learned_reranker.config import L1TrainConfig, L2TrainConfig
from learned_reranker.l1 import fit_pairwise_linear_ranker, recall_at_percentile
from learned_reranker.l2 import hash_split, relevance_bins, sample_groups
from learned_reranker.schema import L1_FEATURES
from learned_reranker.training import load_labels


def test_pairwise_linear_ranker_learns_basic_direction() -> None:
    x = np.array(
        [
            [0.10, 0.20, 0.10, 0.30, 0.00, 0.20, 0.80],
            [0.55, 0.45, 0.60, 0.70, 1.00, 0.55, 0.93],
            [0.90, 0.85, 0.95, 0.95, 1.00, 0.85, 0.99],
        ],
        dtype=float,
    )
    y = np.array([0.10, 0.60, 0.95])
    config = L1TrainConfig(epochs=30, max_pairs=10)
    model = fit_pairwise_linear_ranker(x, y, alpha=0.01, gamma=1.0, config=config)
    z = model.z(x, np.zeros(len(L1_FEATURES)), mu=0.0)
    assert z[-1] > z[0]
    assert recall_at_percentile(z, y, 34) >= 0.5


def test_relevance_bins_are_8_bins() -> None:
    y = np.array([0.0, 0.1, 0.5, 0.99, 1.0])
    assert relevance_bins(y, bins=8).tolist() == [0, 0, 4, 7, 7]


def test_load_labels_accepts_human_and_llm_column_names(tmp_path) -> None:
    human_path = tmp_path / "human.csv"
    human_path.write_text("doc_id,score\nabc,1.2\ndef,-0.1\n", encoding="utf-8")
    human = load_labels(human_path)
    assert human.columns == ["doc_id", "gold_score"]
    assert human["gold_score"].to_list() == [1.0, 0.0]

    llm_path = tmp_path / "llm.csv"
    llm_path.write_text("id,quality_score\nabc,0.75\n", encoding="utf-8")
    llm = load_labels(llm_path)
    assert llm.row(0, named=True) == {"doc_id": "abc", "gold_score": 0.75}


def test_hash_split_is_stable_and_grouping_uses_content_type_length() -> None:
    ids = ["a", "b", "c"]
    assert hash_split(ids).train.tolist() == hash_split(ids).train.tolist()
    df = pl.DataFrame(
        {
            "doc_id": [str(i) for i in range(10)],
            "content_type": ["report"] * 10,
            "length": [1] * 10,
        }
    )
    groups = sample_groups(df, L2TrainConfig(group_size=4, groups_per_bucket=2))
    assert len(groups) == 2
    assert all(len(group) == 4 for group in groups)
