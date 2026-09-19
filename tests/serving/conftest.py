"""Fixtures shared by serving tests."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from churn_mlops.serving.api import app
from churn_mlops.serving.model_loader import LoadedModel, ModelMetadata
from churn_mlops.serving.schemas import InputFeatures


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_input() -> InputFeatures:
    return InputFeatures(
        age=45,
        tenure=24,
        usage_frequency=12,
        support_calls=1,
        payment_delay=0,
        last_interaction=5,
        total_spend=800.0,
        gender="Female",
        subscription_type="Premium",
        contract_length="Annual",
    )


@pytest.fixture
def sample_json(sample_input: InputFeatures) -> dict[str, Any]:
    return sample_input.model_dump()


@pytest.fixture
def mock_model_factory() -> Callable[..., LoadedModel]:
    def factory(proba: float = 0.5, threshold: float = 0.5) -> LoadedModel:
        class MockModel:
            def predict_proba(self, X):
                return np.array([[1 - proba, proba]])

        metadata = ModelMetadata(
            model_name="churn-propensity",
            model_alias="champion",
            model_version=5,
            threshold=threshold,
        )

        return LoadedModel(
            model=MockModel(),
            metadata=metadata,
        )

    return factory
