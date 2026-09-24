"""Fixtures shared by tracking tests."""

from unittest.mock import Mock

import pytest
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
        metadata={
            "train_rows": 10,
            "test_rows": 5,
            "feature_count": 3,
            "feature_names_in": ["a", "b"],
            "feature_names_out": ["a", "b", "c"],
        },
    )
