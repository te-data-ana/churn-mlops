"""Fixtures shared by serving tests."""

from collections.abc import Callable, Generator
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from churn_mlops.serving import api
from churn_mlops.serving.model_loader import LoadedModel, ModelMetadata
from churn_mlops.serving.predictor import PredictionResult
from churn_mlops.serving.schemas import (
    InputFeatures,
    PredictionErrorEvent,
    PredictionEvent,
)


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    mock_model_factory: Callable[..., LoadedModel],
) -> Generator[Any, Any, Any]:
    monkeypatch.setattr(api, "load_model", mock_model_factory)
    with TestClient(api.app) as test_client:
        yield test_client


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
def sample_input_factory() -> Callable[..., dict]:
    def factory(**kwargs) -> dict:
        data = {
            "age": 45,
            "tenure": 24,
            "usage_frequency": 12,
            "support_calls": 1,
            "payment_delay": 0,
            "last_interaction": 5,
            "total_spend": 800.0,
            "gender": "Female",
            "subscription_type": "Premium",
            "contract_length": "Annual",
        }
        data.update(kwargs)

        return data

    return factory


@pytest.fixture
def sample_dict(sample_input: InputFeatures) -> dict[str, Any]:
    return sample_input.model_dump()


@pytest.fixture
def sample_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_name="churn-propensity",
        model_alias="champion",
        model_version=5,
        threshold=0.5,
    )


@pytest.fixture
def sample_prediction_result(sample_metadata) -> PredictionResult:
    return PredictionResult(
        predicted_class=1, predicted_probability=0.87, metadata=sample_metadata
    )


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


@pytest.fixture
def sample_prediction_event() -> PredictionEvent:
    return PredictionEvent(
        request_id="req-1",
        timestamp_utc=datetime.now(UTC),
        latency_ms=10.0,
        features=None,
        model_name="model",
        model_alias="champion",
        model_version=1,
        predicted_class=1,
        predicted_probability=0.9,
        threshold=0.5,
    )


@pytest.fixture
def sample_prediction_error_event() -> PredictionErrorEvent:
    exc = ValueError("invalid input")

    return PredictionErrorEvent(
        request_id="req-1",
        timestamp_utc=datetime.now(UTC),
        latency_ms=10.0,
        error_type=type(exc).__name__,
        error_message=exc.args[0],
        traceback="sample traceback",
    )


@pytest.fixture
def prediction_event_factory() -> Callable[..., PredictionEvent]:
    def factory(**kwargs) -> PredictionEvent:
        return PredictionEvent(
            request_id=kwargs.get("request_id", "req-1"),
            timestamp_utc=kwargs.get("timestamp_utc", datetime.now(UTC)),
            latency_ms=kwargs.get("latency_ms", 10.0),
            features=kwargs.get("features"),
            model_name=kwargs.get("model_name", "model"),
            model_alias=kwargs.get("model_alias", "champion"),
            model_version=kwargs.get("model_version", 1),
            predicted_class=kwargs.get("predicted_class", 1),
            predicted_probability=kwargs.get("predicted_probability", 0.9),
            threshold=kwargs.get("threshold", 0.5),
        )

    return factory
