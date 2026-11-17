from collections.abc import Callable

import pytest
from pydantic import ValidationError

from churn_mlops.serving.schemas import (
    PredictionErrorEvent,
    PredictionEvent,
)


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
