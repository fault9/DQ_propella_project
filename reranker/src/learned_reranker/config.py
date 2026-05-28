from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from learned_reranker.schema import PRIORITY_FEATURES


class L2Mode(str, Enum):
    disabled = "disabled"
    prototype = "prototype"
    trained = "trained"


class OutputFormat(str, Enum):
    parquet = "parquet"
    jsonl = "jsonl"


class HardFilter(BaseModel):
    expected_lid: str = "swe_Latn"
    min_content_quality: int | None = None
    min_information_density: int | None = None
    min_educational_value: int | None = None
    min_safety: int | None = None
    require_pii_absence: bool | None = None
    min_length: int | None = None
    min_language_confidence: float | None = None


class DatasetConfig(BaseModel):
    propella_dataset: str = "openeurollm/propella-annotations"
    propella_config: str = "swe_Latn"
    propella_source_dataset: str = "finepdfs"
    finepdf_dataset: str = "HuggingFaceFW/finepdfs"
    finepdf_config: str = "swe_Latn"
    split: str = "train"
    limit: int | None = 10_000


class PriorityConfig(BaseModel):
    credits: dict[str, float] = Field(
        default_factory=lambda: {
            name: 100.0 / len(PRIORITY_FEATURES) for name in PRIORITY_FEATURES
        }
    )
    mu: float | None = None

    @field_validator("credits")
    @classmethod
    def validate_credits(cls, value: dict[str, float]) -> dict[str, float]:
        missing = set(PRIORITY_FEATURES) - set(value)
        extra = set(value) - set(PRIORITY_FEATURES)
        if missing:
            raise ValueError(f"Missing priority credits for: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unknown priority features: {sorted(extra)}")
        total = sum(value.values())
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"Priority credits must sum to 100, got {total}")
        if any(v < 0 for v in value.values()):
            raise ValueError("Priority credits must be non-negative")
        return value

    def as_unit_vector(self) -> dict[str, float]:
        return {name: credit / 100.0 for name, credit in self.credits.items()}


class RankingConfig(BaseModel):
    k: int = 100
    m: int = 10
    l2_mode: L2Mode = L2Mode.prototype
    output_format: OutputFormat = OutputFormat.parquet
    output_path: Path = Path("outputs/reranked.parquet")
    include_filtered_rows: bool = True

    @model_validator(mode="after")
    def validate_budget(self) -> "RankingConfig":
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.m < 1:
            raise ValueError("m must be at least 1")
        return self


class PrototypeConfig(BaseModel):
    enabled_warning: str = (
        "PROTOTYPE scoring is active. Replace with trained artifacts once gold labels exist."
    )
    l1_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "content_quality": 1.25,
            "information_density": 1.05,
            "educational_value": 1.25,
            "safety": 1.15,
            "pii_absence": 0.95,
            "length": 0.70,
            "language_confidence": 0.95,
        }
    )
    l2_extra_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "inverted_cluster_size": 0.35,
            "inverted_duplicate_count": 0.35,
        }
    )


class L1TrainConfig(BaseModel):
    alpha_grid: list[float] = Field(default_factory=lambda: [1e-4, 1e-2, 1.0, 100.0, 1e4])
    gamma_grid: list[float] = Field(default_factory=lambda: [0.0, 1.0, 2.0, 5.0, 10.0])
    learning_rate: float = 0.05
    epochs: int = 250
    max_pairs: int = 250_000
    eval_percentiles: list[int] = Field(default_factory=lambda: [1, 5, 10, 20, 30])


class L2TrainConfig(BaseModel):
    relevance_bins: int = 8
    eval_k: list[int] = Field(default_factory=lambda: [10, 25, 50, 100])
    group_size: int = 64
    groups_per_bucket: int = 32
    lightgbm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_data_in_leaf": 30,
            "num_leaves": 15,
            "verbose": -1,
        }
    )

    @field_validator("relevance_bins")
    @classmethod
    def validate_bins(cls, value: int) -> int:
        if value != 8:
            raise ValueError("This prototype intentionally uses 8 LTR relevance bins")
        return value


class PipelineConfig(BaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    hard_filter: HardFilter = Field(default_factory=HardFilter)
    priority: PriorityConfig = Field(default_factory=PriorityConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    prototype: PrototypeConfig = Field(default_factory=PrototypeConfig)
    l1_train: L1TrainConfig = Field(default_factory=L1TrainConfig)
    l2_train: L2TrainConfig = Field(default_factory=L2TrainConfig)
    l1_model_path: Path | None = None
    l2_model_path: Path | None = None


def load_config(path: Path) -> PipelineConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PipelineConfig.model_validate(data)


def save_config(config: PipelineConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json")
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
