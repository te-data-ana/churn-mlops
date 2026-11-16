from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from churn_mlops.config import load_config
from churn_mlops.config.schemas import (
    PromotionConfig,
    ThresholdConfig,
    TrainingConfig,
)


@pytest.mark.unit
def test_load_config_returns_config_and_resolved_path(sample_config_yaml: Path) -> None:
    config_file_name = sample_config_yaml.name
    config_file_dir = sample_config_yaml.parent

    config, path = load_config(
        file=config_file_name,
        path=config_file_dir,
    )

    assert config.use_case == "churn"
    assert config.tuning.n_trials == 50
    assert config.calibration.n_splits == 3
    assert config.threshold.strategy == "max_f1"
    assert sum(config.promotion.metric_weights.values()) > 0
    assert path == sample_config_yaml


@pytest.mark.unit
@pytest.mark.parametrize(
    "filename",
    [
        "config.yaml",
        "dc.yaml",
        "dt.yaml",
        "hgb.yaml",
        "lr.yaml",
        "nb.yaml",
        "rf.yaml",
        "sample_training_config.yaml",
    ],
)
def test_shipped_training_configs_have_valid_settings(
    filename: str,
) -> None:
    config_directory = Path(__file__).resolve().parents[2] / "src" / "config"

    config, _ = load_config(file=filename, path=config_directory)

    assert config.tuning.n_trials >= 1
    assert config.calibration.n_splits >= 2
    assert sum(config.promotion.metric_weights.values()) > 0


@pytest.mark.unit
def test_promotion_config_rejects_negative_min_delta() -> None:
    with pytest.raises(ValueError, match="min_delta"):
        PromotionConfig(min_delta=-0.1)


@pytest.mark.unit
def test_training_config_rejects_empty_use_case(
    config_factory: Callable[..., TrainingConfig],
) -> None:
    with pytest.raises(ValueError, match="use_case"):
        replace(config_factory(), use_case="")


@pytest.mark.unit
def test_threshold_config_rejects_unsupported_strategy() -> None:
    with pytest.raises(ValueError, match="threshold strategy"):
        ThresholdConfig(strategy="youden")
