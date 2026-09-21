from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from churn_mlops.serving.model_loader import LoadedModel
from churn_mlops.serving.predictor import PredictionResult, Predictor
from churn_mlops.serving.schemas import InputFeatures


@pytest.mark.unit
def test_predict_returns_probability_class_and_metadata(
    mock_model_factory: Callable[..., LoadedModel], sample_input: InputFeatures
) -> None:
    loaded_model = mock_model_factory(proba=0.8, threshold=0.5)

    result = Predictor(loaded_model).predict(sample_input)

    assert isinstance(result, PredictionResult)
    assert result.predicted_class == 1
    assert result.predicted_probability == 0.8
    assert result.metadata == loaded_model.metadata
    assert result.metadata.model_name == "churn-propensity"
    assert result.metadata.model_alias == "champion"
    assert result.metadata.threshold == 0.5


@pytest.mark.unit
@pytest.mark.parametrize(
    ("probability", "threshold", "expected_class"),
    [
        (0.4, 0.5, 0),
        (0.5, 0.5, 1),
        (0.7, 0.5, 1),
    ],
)
def test_predict_classifies_depending_on_threshold(
    probability: float,
    threshold: float,
    expected_class: int,
    mock_model_factory: Callable[..., LoadedModel],
    sample_input: InputFeatures,
) -> None:
    loaded_model = mock_model_factory(proba=probability, threshold=threshold)

    result = Predictor(loaded_model).predict(sample_input)

    assert result.predicted_class == expected_class


@pytest.mark.unit
def test_predict_batch_returns_predictions_and_preserves_index(
    mock_model_factory: Callable[..., LoadedModel],
) -> None:
    # Use two rows and a non-default index to verify batch mapping and index retention.
    loaded_model = mock_model_factory(proba=0.5, threshold=0.5)
    loaded_model.model.predict_proba = Mock(
        return_value=np.array([[0.8, 0.2], [0.3, 0.7]])
    )
    predictor = Predictor(loaded_model)
    input_df = pd.DataFrame({"feature": [10, 20]}, index=[101, 205])

    result = predictor.predict_batch(input_df)

    assert list(result.index) == [101, 205]
    assert list(result["predicted_probability"]) == [0.2, 0.7]
    assert list(result["predicted_class"]) == [0, 1]
    assert list(result["threshold"]) == [0.5, 0.5]
    assert list(result["model_version"]) == [5, 5]
