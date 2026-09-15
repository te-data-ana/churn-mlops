from unittest.mock import MagicMock

import pytest
from mlflow.exceptions import MlflowException

import churn_mlops.tracking.mlflow as tracking
from churn_mlops.config import TMP_DIR


@pytest.mark.unit
def test_setup_local_experiment_creates_new(monkeypatch):
    monkeypatch.setattr(
        tracking.mlflow, "create_experiment", lambda *args, **kwargs: "123"
    )

    mock_set_experiment = MagicMock()

    monkeypatch.setattr(tracking.mlflow, "set_experiment", mock_set_experiment)

    experiment_id = tracking.setup_local_experiment("test")

    assert experiment_id == "123"
    mock_set_experiment.assert_called_once()


@pytest.mark.unit
def test_setup_local_experiment_existing(monkeypatch):

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
def test_log_experiment_result(
    monkeypatch,
    mock_taining_result,
    config_factory,
):
    config_file_path = TMP_DIR / "test.yaml"
    config_file_path.write_text("test")

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
        mock_taining_result, config_factory(), config_file_path
    )

    mock_log_metrics.assert_called_once()
    mock_log_param.assert_called_once()
    assert mock_log_params.call_count > 4
    mock_set_tags.assert_called_once()
    mock_log_model.assert_called_once()
    assert mock_log_artifact.call_count >= 2
