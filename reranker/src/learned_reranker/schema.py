from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    higher_is_better: bool
    user_priority: bool
    l1_feature: bool
    l2_feature: bool


PROPELLA_FEATURES: tuple[str, ...] = (
    "content_quality",
    "information_density",
    "educational_value",
    "safety",
    "pii_absence",
    "length",
)

FINEPDF_PRIORITY_FEATURES: tuple[str, ...] = ("language_confidence",)

PRIORITY_FEATURES: tuple[str, ...] = PROPELLA_FEATURES + FINEPDF_PRIORITY_FEATURES

L1_FEATURES: tuple[str, ...] = PRIORITY_FEATURES

L2_EXTRA_FEATURES: tuple[str, ...] = (
    "inverted_cluster_size",
    "inverted_duplicate_count",
)

L2_FEATURES: tuple[str, ...] = L1_FEATURES + ("z_l1",) + tuple(
    f"ux_{name}" for name in PRIORITY_FEATURES
) + L2_EXTRA_FEATURES

RAW_FINEPDF_COLUMNS: tuple[str, ...] = (
    "id",
    "full_doc_lid",
    "full_doc_lid_score",
    "minhash_cluster_size",
    "duplicate_count",
)

RAW_PROPELLA_COLUMNS: tuple[str, ...] = (
    "id",
    "content_quality",
    "information_density",
    "educational_value",
    "safety",
    "pii_presence",
    "content_length",
    "content_type",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "passed_hard_filter",
    "rank_l1",
    "z_l1",
    "rank_l2",
    "score_l2",
    "in_top_k",
    "minhash_cluster_size",
    "duplicate_count",
)
