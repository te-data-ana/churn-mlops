"""Fixtures shared by tracking tests."""

from unittest.mock import Mock

import pytest
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline

from churn_mlops.models.features import FeatureBuilder
from churn_mlops.tracking.registry import ModelRegistry
from churn_mlops.training import TrainingResult


@pytest.fixture
def registry() -> ModelRegistry:
    registry = ModelRegistry()
    registry.client = Mock()
    return registry


@pytest.fixture
def mock_training_result() -> TrainingResult:
    return TrainingResult(
        trained_pipeline=Pipeline(steps=[("features", FeatureBuilder())]),
        metrics={},
        classifier_config={},
        feature_names_in=["a", "b"],
        feature_names_out=["a", "b", "c"],
        threshold=0.4,
        threshold_data="out_of_time_test",
        calibration_figure=Figure(),
    )
