from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from churn_mlops.serving.schemas import (
    InputFeatures,
    PredictionErrorEvent,
    PredictionEvent,
)


@pytest.mark.unit
def test_input_features_accepts_valid_values(
    sample_input: InputFeatures,
) -> None:
    assert sample_input.age == 45
    assert sample_input.gender == "Female"
    assert sample_input.subscription_type == "Premium"


def test_input_features_accepts_boundary_values() -> None:
    input_features = InputFeatures(
        age=115,
        tenure=0,
        usage_frequency=0,
        support_calls=0,
        payment_delay=0,
        last_interaction=0,
        total_spend=0.0,
        gender="Male",
        subscription_type="Basic",
        contract_length="Monthly",
    )

    assert input_features.age == 115
    assert input_features.total_spend == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", 17),
        ("age", 116),
        ("support_calls", -1),
        ("payment_delay", -1),
        ("gender", "XY"),
        ("subscription_type", "Free"),
        ("contract_length", "undefined"),
    ],
)
def test_input_features_rejects_invalid_values(
    sample_input_factory: Callable[..., dict],
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        InputFeatures(**sample_input_factory(**{field: value}))


def test_prediction_event_has_default_event_name(
    sample_prediction_event: PredictionEvent,
) -> None:
    prediction_event = sample_prediction_event

    assert prediction_event.event == "prediction"


def test_prediction_error_event_has_default_event_name(
    sample_prediction_error_event: PredictionErrorEvent,
) -> None:
    prediction_error_event = sample_prediction_error_event

    assert prediction_error_event.event == "prediction_error"


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01],
)
def test_prediction_event_rejects_invalid_probability(
    prediction_event_factory: Callable[..., PredictionEvent],
    probability: float,
) -> None:
    with pytest.raises(ValidationError):
        prediction_event_factory(
            predicted_probability=probability,
        )


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1],
)
def test_prediction_event_rejects_invalid_threshold(
    prediction_event_factory: Callable[..., PredictionEvent],
    threshold: float,
) -> None:
    with pytest.raises(ValidationError):
        prediction_event_factory(
            threshold=threshold,
        )


def test_prediction_event_is_immutable(
    sample_prediction_error_event: PredictionErrorEvent,
) -> None:
    event = sample_prediction_error_event

    with pytest.raises(ValidationError):
        event.request_id = "new-id"


def test_prediction_event_allows_missing_features(
    sample_prediction_error_event: PredictionErrorEvent,
) -> None:
    event = sample_prediction_error_event

    assert event.features is None
