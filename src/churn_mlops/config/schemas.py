from dataclasses import dataclass, field
from math import isfinite
from typing import Any


@dataclass
class ClassifierConfig:
    classifier: str
    classifier_params: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate classifier configuration."""
        if not isinstance(self.classifier, str) or not self.classifier:
            raise ValueError("classifier alias must be provided as a string.")
        if not isinstance(self.classifier_params, dict):
            raise TypeError("classifier_params must be a dictionary.")


@dataclass
class DataConfig:
    target_column: str = "target"
    time_column: str = "reference_date"

    def __post_init__(self) -> None:
        """Validate column names."""
        if not isinstance(self.target_column, str) or not self.target_column:
            raise ValueError("target_column name must be provided as a string.")
        if not isinstance(self.time_column, str) or not self.time_column:
            raise ValueError("time_column name must be provided as a string.")


@dataclass
class FeatureBuilderConfig:
    feature_params: dict[str, int]


@dataclass
class PreprocessingConfig:
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool = False
    n_trials: int = 50
    n_splits: int = 4
    metric: str = "roc_auc"
    random_state: int = 26


@dataclass
class CalibrationConfig:
    method: str = "none"
    n_splits: int = 3
    n_bins: int = 10

    def __post_init__(self) -> None:
        """Validate calibration settings."""
        if self.method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError(
                "calibration method must be one of 'none', 'sigmoid', or 'isotonic'."
            )
        if not isinstance(self.n_splits, int) or self.n_splits < 2:
            raise ValueError("calibration n_splits must be at least 2.")
        if not isinstance(self.n_bins, int) or self.n_bins < 1:
            raise ValueError("calibration n_bins must be a positive integer.")


@dataclass
class ThresholdConfig:
    strategy: str = "max_f1"

    def __post_init__(self) -> None:
        """Validate threshold selection settings."""
        if self.strategy != "max_f1":
            raise ValueError("threshold strategy currently supports only 'max_f1'.")


@dataclass
class RegistryConfig:
    register_model: bool = False
    model_name: str = "churn-risk"
    alias: str = "candidate"


@dataclass
class PromotionConfig:
    promote_model: bool = False
    min_delta: float = 0.005
    metric_weights: dict[str, float] = field(
        default_factory=lambda: {
            "roc_auc": 0.2,
            "pr_auc": 0.2,
            "f1": 0.2,
            "brier_score": 0.2,
            "calibration_error": 0.2,
        }
    )

    def __post_init__(self) -> None:
        """Validate promotion weights and the required promotion delta."""
        allowed_metrics = {
            "roc_auc",
            "pr_auc",
            "f1",
            "brier_score",
            "calibration_error",
        }
        if (
            not isinstance(self.metric_weights, dict)
            or not self.metric_weights
            or not set(self.metric_weights).issubset(allowed_metrics)
        ):
            raise ValueError(
                "metric_weights must include one or more supported evaluation "
                f"metrics: {', '.join(sorted(allowed_metrics))}."
            )
        if any(
            not isinstance(weight, float) or not isfinite(weight) or weight < 0
            for weight in self.metric_weights.values()
        ):
            raise ValueError("metric_weights must be finite and non-negative.")
        if sum(self.metric_weights.values()) <= 0:
            raise ValueError("metric_weights must have a positive total weight.")
        if (
            not isinstance(self.min_delta, float)
            or not isfinite(self.min_delta)
            or self.min_delta < 0
        ):
            raise ValueError("min_delta must be finite and non-negative.")


@dataclass(frozen=True)
class TrainingConfig:
    model: ClassifierConfig
    data: DataConfig
    feature_builder: FeatureBuilderConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
