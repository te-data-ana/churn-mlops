from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_mlops import training
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
@pytest.mark.parametrize(
    ("provided_name", "configured_name", "expected_name"),
    [
        (None, "configured-experiment", "configured-experiment"),
        ("explicit-experiment", "configured-experiment", "explicit-experiment"),
    ],
)
def test_run_training_job_resolves_mlflow_experiment_name(
    monkeypatch: pytest.MonkeyPatch,
    provided_name: str | None,
    configured_name: str,
    expected_name: str,
) -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(classifier="lr"),
        evaluation=SimpleNamespace(threshold=0.5),
        registry=SimpleNamespace(register_model=False),
    )
    settings = SimpleNamespace(
        config_dir=Path("config"),
        raw_data_dir=Path("data"),
        artifact_dir=Path("artifacts"),
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment_name=configured_name,
    )
    training_result = MagicMock()
    setup_experiment = MagicMock(return_value="experiment-id")
    mlflow_run = MagicMock()

    monkeypatch.setattr(training, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(
        training, "load_config", lambda **_: (config, Path("training.yaml"))
    )
    monkeypatch.setattr(training, "load_raw_data", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(training, "validate_training_data", lambda data: data)
    monkeypatch.setattr(training, "train", MagicMock(return_value=training_result))
    monkeypatch.setattr(training, "setup_local_experiment", setup_experiment)
    monkeypatch.setattr(training, "log_experiment_result", MagicMock())
    monkeypatch.setattr(training.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(
        training.mlflow, "start_run", MagicMock(return_value=mlflow_run)
    )

    training.run_training_job(experiment_name=provided_name)

    assert setup_experiment.call_args.kwargs["experiment_name"] == expected_name


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
