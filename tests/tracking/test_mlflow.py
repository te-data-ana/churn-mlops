import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

import churn_mlops.tracking.mlflow as tracking
from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.training import TrainingResult, train_model


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
    sample_config_yaml: Path,
    mock_training_result: TrainingResult,
    config_factory: Callable[..., TrainingConfig],
) -> None:

    # Mock MLflow logging components and verify each category is logged (at least) once.
    mock_log_metrics = MagicMock()
    mock_log_param = MagicMock()
    mock_log_params = MagicMock()
    mock_set_tags = MagicMock()
    mock_log_model = MagicMock()
    mock_log_artifact = MagicMock()
    mock_log_figure = MagicMock()

    monkeypatch.setattr(tracking.mlflow, "log_metrics", mock_log_metrics)
    monkeypatch.setattr(tracking.mlflow, "log_param", mock_log_param)
    monkeypatch.setattr(tracking.mlflow, "log_params", mock_log_params)
    monkeypatch.setattr(tracking.mlflow, "set_tags", mock_set_tags)
    monkeypatch.setattr(tracking.mlflow.sklearn, "log_model", mock_log_model)
    monkeypatch.setattr(tracking.mlflow, "log_artifact", mock_log_artifact)
    monkeypatch.setattr(tracking.mlflow, "log_figure", mock_log_figure)

    mock_training_result.classifier_config["random_state"] = 42
    tracking.log_experiment_result(
        mock_training_result, config_factory(), sample_config_yaml
    )

    mock_log_metrics.assert_called_once()
    mock_log_param.assert_called_once_with("feature_count", 3)
    assert mock_log_params.call_count > 4
    logged_parameters = {
        name for call in mock_log_params.call_args_list for name in call.args[0]
    }
    assert "test_size" not in logged_parameters
    assert "data_random_state" not in logged_parameters
    assert "random_state" in logged_parameters
    assert "threshold" in logged_parameters
    assert "calibration_method" in logged_parameters
    assert "threshold_strategy" in logged_parameters
    assert "threshold_data" in logged_parameters
    mock_set_tags.assert_called_once()
    mock_log_model.assert_called_once()
    assert mock_log_artifact.call_count >= 2
    mock_log_figure.assert_called_once()
    assert any(
        call.kwargs.get("artifact_path") == "evaluation"
        for call in mock_log_artifact.call_args_list
    )


@pytest.mark.integration
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_logged_calibrated_model_can_be_reloaded(
    method: str,
    tmp_path: Path,
    sample_config_yaml: Path,
    sample_training_df: pd.DataFrame,
    config_factory: Callable[..., TrainingConfig],
    mlflow_test_setup: dict[str, object],
) -> None:
    config = config_factory(classifier="lr")
    config.calibration.method = method
    train_df = pd.concat([sample_training_df] * 8, ignore_index=True)
    train_df["reference_date"] = pd.date_range(
        "2026-01-01",
        periods=len(train_df),
    )
    test_df = sample_training_df.copy()
    test_df["reference_date"] = pd.date_range(
        "2026-03-22",
        periods=len(test_df),
    )
    result = train_model(config, train_df, test_df)
    config_path = tmp_path / f"{method}.yaml"
    config_path.write_text(
        sample_config_yaml.read_text(encoding="utf-8").replace(
            "method: none",
            f"method: {method}",
        ),
        encoding="utf-8",
    )
    experiment_id = tracking.setup_local_experiment(
        experiment_name=f"calibration-{method}",
        tracking_uri=str(mlflow_test_setup["tracking_uri"]),
        artifact_dir=tmp_path / "artifacts",
    )

    with mlflow.start_run(experiment_id=experiment_id):
        model_info = tracking.log_experiment_result(
            result=result,
            config=config,
            config_file_path=config_path,
            artifact_dir=tmp_path / "artifacts",
        )
        reloaded_model = mlflow.sklearn.load_model(model_info.model_uri)

    client = MlflowClient()
    calibration_artifacts = client.list_artifacts(
        model_info.run_id,
        "calibration",
    )
    threshold_artifacts = client.list_artifacts(model_info.run_id, "evaluation")
    threshold_path = client.download_artifacts(
        model_info.run_id,
        "evaluation/decision_threshold.json",
        str(tmp_path),
    )
    threshold_metadata = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
    logged_run = client.get_run(model_info.run_id)
    probabilities = reloaded_model.predict_proba(
        test_df.drop(columns=["churn", "reference_date"])
    )

    assert probabilities.shape == (len(test_df), 2)
    assert any(
        item.path.endswith("reliability_diagram.png") for item in calibration_artifacts
    )
    assert any(
        item.path.endswith("decision_threshold.json") for item in threshold_artifacts
    )
    assert threshold_metadata["threshold_data"] == "out_of_time_test"
    assert set(threshold_metadata) == {
        "calibration_method",
        "threshold_strategy",
        "threshold_data",
        "threshold",
    }
    assert logged_run.data.params["threshold_data"] == "out_of_time_test"
