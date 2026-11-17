import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import pandas as pd
import pytest
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.calibration import CalibratedClassifierCV

import churn_mlops.tracking.mlflow as tracking
from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.models import build_classifier_pipeline, create_model
from churn_mlops.models.features import IdentityFeatureBuilder
from churn_mlops.training import TrainingResult
from churn_mlops.use_cases import UseCase, get_use_case


@pytest.mark.unit
def test_setup_local_experiment_returns_new_experiment_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        tracking.mlflow, "create_experiment", lambda *args, **kwargs: "123"
    )

    mock_set_experiment = MagicMock()
    monkeypatch.setattr(tracking.mlflow, "set_experiment", mock_set_experiment)

    experiment_id = tracking.setup_local_experiment(
        "test",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_dir=tmp_path / "artifacts",
    )

    assert experiment_id == "123"
    mock_set_experiment.assert_called_once()


@pytest.mark.unit
def test_setup_local_experiment_returns_existing_experiment_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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

    experiment_id = tracking.setup_local_experiment(
        "test",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        artifact_dir=tmp_path / "artifacts",
    )

    assert experiment_id == "123"


@pytest.mark.unit
def test_log_experiment_result_logs_metrics_parameters_model_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    sample_config_yaml: Path,
    mock_training_result: TrainingResult,
    config_factory: Callable[..., TrainingConfig],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)

    # Mock MLflow logging components and verify each category is logged (at least) once.
    mock_log_metrics = MagicMock()
    mock_log_params = MagicMock()
    mock_set_tags = MagicMock()
    mock_log_model = MagicMock()
    mock_log_artifact = MagicMock()
    mock_log_figure = MagicMock()

    monkeypatch.setattr(tracking.mlflow, "log_metrics", mock_log_metrics)
    monkeypatch.setattr(tracking.mlflow, "log_params", mock_log_params)
    monkeypatch.setattr(tracking.mlflow, "set_tags", mock_set_tags)
    monkeypatch.setattr(tracking.mlflow.sklearn, "log_model", mock_log_model)
    monkeypatch.setattr(tracking.mlflow, "log_artifact", mock_log_artifact)
    monkeypatch.setattr(tracking.mlflow, "log_figure", mock_log_figure)

    mock_training_result.classifier_config["random_state"] = 42
    tracking.log_experiment_result(
        use_case=get_use_case("churn"),
        result=mock_training_result,
        config=config_factory(),
        config_file_path=sample_config_yaml,
        calibration_method="sigmoid",
        tune_model=True,
        register_model=True,
        promote_model=True,
    )

    mock_log_metrics.assert_called_once()
    assert mock_log_params.call_count > 4
    logged_parameters = {
        name for call in mock_log_params.call_args_list for name in call.args[0]
    }
    logged_parameter_values = {
        name: value
        for call in mock_log_params.call_args_list
        for name, value in call.args[0].items()
    }
    assert "test_size" not in logged_parameters
    assert "data_random_state" not in logged_parameters
    assert "random_state" in logged_parameters
    assert logged_parameter_values["use_case"] == "churn"
    assert "threshold" in logged_parameters
    assert "calibration_method" in logged_parameters
    assert "model_tuning_enabled" in logged_parameters
    assert "model_registration_enabled" in logged_parameters
    assert "model_promotion_enabled" in logged_parameters
    assert logged_parameter_values["calibration_method"] == "sigmoid"
    assert logged_parameter_values["model_tuning_enabled"] is True
    assert logged_parameter_values["model_registration_enabled"] is True
    assert logged_parameter_values["model_promotion_enabled"] is True
    assert "threshold_strategy" in logged_parameters
    assert "threshold_data" in logged_parameters
    mock_set_tags.assert_called_once()
    mock_log_model.assert_called_once()
    mock_log_figure.assert_called_once_with(
        mock_training_result.calibration_figure,
        artifact_file="calibration/reliability_diagram.png",
    )
    assert mock_log_artifact.call_count >= 1
    assert "Logging training parameters and tags to MLflow." in caplog.text
    assert "Logging evaluation metrics to MLflow:" in caplog.text
    assert "Logging calibration reliability diagram to MLflow." in caplog.text
    assert "Logging fitted model pipeline to MLflow." in caplog.text
    assert "Logging training configuration artifact" in caplog.text


@pytest.mark.unit
def test_log_experiment_result_trusts_selected_use_case_feature_builder(
    monkeypatch: pytest.MonkeyPatch,
    sample_config_yaml: Path,
    mock_training_result: TrainingResult,
    config_factory: Callable[..., TrainingConfig],
) -> None:
    feature_builder_factory = MagicMock(return_value=IdentityFeatureBuilder())
    use_case = UseCase(
        name="alternate",
        target_column="label",
        time_column="event_date",
        feature_columns=("value",),
        data_validator=lambda frame: frame,
        feature_builder_factory=feature_builder_factory,
    )
    config = config_factory()
    mock_training_result.trained_pipeline.steps[0] = (
        "features",
        IdentityFeatureBuilder(),
    )
    mock_log_model = MagicMock()

    monkeypatch.setattr(tracking.mlflow, "log_params", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "log_metrics", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "log_figure", MagicMock())
    monkeypatch.setattr(tracking.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr(tracking.mlflow.sklearn, "log_model", mock_log_model)

    tracking.log_experiment_result(
        use_case=use_case,
        result=mock_training_result,
        config=config,
        config_file_path=sample_config_yaml,
        calibration_method="none",
        tune_model=False,
        register_model=False,
        promote_model=False,
    )

    expected_trusted_type = (
        f"{IdentityFeatureBuilder.__module__}.{IdentityFeatureBuilder.__name__}"
    )
    trusted_types = mock_log_model.call_args.kwargs["skops_trusted_types"]
    assert expected_trusted_type in trusted_types
    assert not any(
        "ChurnFeatureBuilder" in trusted_type for trusted_type in trusted_types
    )
    feature_builder_factory.assert_called_once()


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
    use_case = get_use_case("churn")
    config = config_factory(classifier="lr")
    config.calibration.n_splits = 2
    train_df = sample_training_df.copy()
    train_df["reference_date"] = pd.date_range(
        "2026-01-01",
        periods=len(train_df),
    )
    test_df = sample_training_df.copy()
    test_df["reference_date"] = pd.date_range(
        "2026-03-22",
        periods=len(test_df),
    )
    classifier, classifier_config = create_model(
        model_alias=config.model.classifier,
        model_params=config.model.classifier_params,
        n_jobs=1,
    )
    pipeline = build_classifier_pipeline(
        classifier=classifier,
        feature_builder=use_case.build_feature_builder(),
        num_impute_strategy=config.preprocessing.numeric_impute_strategy,
        cat_impute_strategy=config.preprocessing.categorical_impute_strategy,
    )
    fitted_model = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv=config.calibration.n_splits,
        n_jobs=1,
    ).fit(
        X=train_df.loc[:, list(use_case.feature_columns)],
        y=train_df[use_case.target_column],
    )
    fitted_pipeline = fitted_model.calibrated_classifiers_[0].estimator
    result = TrainingResult(
        trained_pipeline=fitted_model,
        metrics={"roc_auc": 0.8},
        classifier_config=classifier_config,
        feature_names_in=list(use_case.feature_columns),
        feature_names_out=(
            fitted_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        ),
        threshold=0.5,
        threshold_data="out_of_time_test",
        calibration_figure=Figure(),
    )
    config_path = tmp_path / f"{method}.yaml"
    config_path.write_text(
        sample_config_yaml.read_text(encoding="utf-8").replace(
            "n_splits: 3",
            "n_splits: 2",
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
            use_case=use_case,
            result=result,
            config=config,
            config_file_path=config_path,
            calibration_method=method,
            tune_model=False,
            register_model=False,
            promote_model=False,
        )
        reloaded_model = mlflow.sklearn.load_model(model_info.model_uri)

    client = MlflowClient(tracking_uri=str(mlflow_test_setup["tracking_uri"]))
    logged_run = client.get_run(model_info.run_id)
    probabilities = reloaded_model.predict_proba(
        test_df.loc[:, list(use_case.feature_columns)]
    )

    assert probabilities.shape == (len(test_df), 2)
    assert logged_run.data.params["threshold_data"] == "out_of_time_test"
