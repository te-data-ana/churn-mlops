from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException

import churn_mlops.tracking.mlflow as tracking
from churn_mlops.config import TMP_DIR
from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.training import TrainingResult


@pytest.mark.unit
def test_setup_local_experiment_returns_new_experiment_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracking.mlflow, "create_experiment", lambda *args, **kwargs: "123"
    )

    mock_set_experiment = MagicMock()
    monkeypatch.setattr(tracking.mlflow, "set_experiment", mock_set_experiment)

    experiment_id = tracking.setup_local_experiment("test")

    assert experiment_id == "123"
    mock_set_experiment.assert_called_once()


@pytest.mark.unit
def test_setup_local_experiment_returns_existing_experiment_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate MLflow reporting that the experiment already exists.
    def raise_exc(*args, **kwargs):
        raise MlflowException("experiment already exists")

    monkeypatch.setattr(tracking.mlflow, "create_experiment", raise_exc)

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "123"
    monkeypatch.setattr(
        tracking.mlflow, "get_experiment_by_name", lambda _: mock_experiment
    )

    mock_set_experiment = MagicMock()
    monkeypatch.setattr(tracking.mlflow, "set_experiment", mock_set_experiment)

    experiment_id = tracking.setup_local_experiment("test")

    assert experiment_id == "123"


@pytest.mark.unit
def test_log_experiment_result_logs_metrics_parameters_model_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    mock_training_result: TrainingResult,
    config_factory: Callable[..., TrainingConfig],
) -> None:
    config_file_path = TMP_DIR / "test.yaml"
    config_file_path.write_text("test")

    # Mock MLflow logging components and verify each category is logged (at least) once.
    mock_log_metrics = MagicMock()
    mock_log_param = MagicMock()
    mock_log_params = MagicMock()
    mock_set_tags = MagicMock()
    mock_log_model = MagicMock()
    mock_log_artifact = MagicMock()

    monkeypatch.setattr(tracking.mlflow, "log_metrics", mock_log_metrics)
    monkeypatch.setattr(tracking.mlflow, "log_param", mock_log_param)
    monkeypatch.setattr(tracking.mlflow, "log_params", mock_log_params)
    monkeypatch.setattr(tracking.mlflow, "set_tags", mock_set_tags)
    monkeypatch.setattr(tracking.mlflow.sklearn, "log_model", mock_log_model)
    monkeypatch.setattr(tracking.mlflow, "log_artifact", mock_log_artifact)

    tracking.log_experiment_result(
        mock_training_result, config_factory(), config_file_path
    )

    mock_log_metrics.assert_called_once()
    mock_log_param.assert_called_once()
    assert mock_log_params.call_count > 4
    mock_set_tags.assert_called_once()
    mock_log_model.assert_called_once()
    assert mock_log_artifact.call_count >= 2
