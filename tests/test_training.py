from collections.abc import Callable
from datetime import datetime
from typing import Literal

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.training import TrainingResult, train


def assert_training_result(result) -> None:
    assert isinstance(result, TrainingResult)
    assert isinstance(result.trained_pipeline, Pipeline)

    assert result.metrics
    assert result.metadata

    metrics = result.metrics
    assert 0.5 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["brier_score"] <= 1
    assert metrics["fit_time_sec"] >= 0

    metadata = result.metadata
    assert datetime.fromisoformat(metadata["timestamp"])
    assert metadata["train_rows"] > 0
    assert metadata["test_rows"] > 0
    assert len(metadata["feature_names_in"]) > 0
    assert len(metadata["feature_names_out"]) > 0


@pytest.mark.integration
def test_run_training_job_returns_valid_training_result(
    registered_model: dict[str, object],
) -> None:
    assert_training_result(registered_model["result"])


@pytest.mark.unit
@pytest.mark.parametrize("classifier", ["lr", "dt", "rf", "hgb"])
def test_train_returns_valid_result_for_supported_classifier(
    classifier: Literal["lr", "dt", "rf", "hgb"],
    config_factory: Callable[..., TrainingConfig],
    sample_training_df: pd.DataFrame,
):
    cfg = config_factory(classifier=classifier)

    result = train(cfg, sample_training_df)

    assert_training_result(result)

    assert len(result.metadata["feature_names_out"]) > len(
        result.metadata["feature_names_in"]
    )


@pytest.mark.unit
@pytest.mark.parametrize("classifier", ["lr", "dt", "rf", "hgb"])
def test_trained_pipeline_predicts_classes_and_probabilities(
    classifier: Literal["lr", "dt", "rf", "hgb"],
    config_factory: Callable[..., TrainingConfig],
    sample_training_df: pd.DataFrame,
) -> None:
    cfg = config_factory(classifier=classifier)
    result = train(cfg, sample_training_df)

    X = sample_training_df.drop(columns=["churn"])

    predictions = result.trained_pipeline.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})

    probas = result.trained_pipeline.predict_proba(X)
    assert probas.shape[0] == len(X)
    assert probas.shape[1] == 2
    assert ((probas >= 0) & (probas <= 1)).all()


@pytest.mark.unit
def test_train_rejects_unknown_classifier(
    config_factory: Callable[..., TrainingConfig],
    sample_training_df: pd.DataFrame,
) -> None:

    cfg = config_factory(classifier="invalid")

    with pytest.raises(ValueError, match="Unknown classifier"):
        train(cfg, sample_training_df)


@pytest.mark.unit
def test_train_returns_reproducible_metrics_for_same_input(
    config_factory: Callable[..., TrainingConfig],
    sample_training_df: pd.DataFrame,
) -> None:
    cfg = config_factory(classifier="dt")

    # Train twice with identical inputs to verify deterministic results.
    result1 = train(cfg, sample_training_df)
    result2 = train(cfg, sample_training_df)

    assert result1.metrics["roc_auc"] == result2.metrics["roc_auc"]
    assert result1.metrics["accuracy"] == result2.metrics["accuracy"]
    assert result1.metadata["train_rows"] == result2.metadata["train_rows"]
