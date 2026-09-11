from dataclasses import dataclass
from typing import Any


@dataclass
class DataConfig:
    target_column: str
    test_size: float
    random_state: int


@dataclass
class FeatureBuilderConfig:
    feature_params: dict[str, int]


@dataclass
class PreprocessingConfig:
    numeric_impute_strategy: str
    categorical_impute_strategy: str


@dataclass
class ClassifierConfig:
    classifier: str
    classifier_params: dict[str, Any]


@dataclass
class EvaluationConfig:
    threshold: float


@dataclass(frozen=True)
class TrainingConfig:
    data: DataConfig
    feature_builder: FeatureBuilderConfig
    preprocessing: PreprocessingConfig
    model: ClassifierConfig
    evaluation: EvaluationConfig
