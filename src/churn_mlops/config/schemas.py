from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataConfig:
    target_column: str


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


@dataclass
class RegistryConfig:
    register_model: bool
    registry_params: dict[str, Any]
    promote_model: bool = True


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool = False
    n_trials: int = 50
    n_splits: int = 4
    time_column: str = "reference_date"
    metric: str = "roc_auc"
    random_state: int = 26


@dataclass(frozen=True)
class TrainingConfig:
    data: DataConfig
    feature_builder: FeatureBuilderConfig
    preprocessing: PreprocessingConfig
    model: ClassifierConfig
    evaluation: EvaluationConfig
    registry: RegistryConfig
    tuning: TuningConfig = field(default_factory=TuningConfig)
