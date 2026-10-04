from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from churn_mlops import train
from churn_mlops.config.schemas import TuningConfig
from churn_mlops.training import TuningResult


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provided_name", "configured_name", "expected_name"),
    [
        (None, "configured-experiment", "configured-experiment"),
        ("explicit-experiment", "configured-experiment", "explicit-experiment"),
    ],
)
def test_run_training_job_resolves_input_arguments(
    monkeypatch: pytest.MonkeyPatch,
    provided_name: str | None,
    configured_name: str,
    expected_name: str,
) -> None:
    settings = SimpleNamespace(
        config_dir=Path("config"),
        data_dir=Path("data"),
        artifact_dir=Path("artifacts"),
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment_name=configured_name,
    )
    config = SimpleNamespace(
        model=SimpleNamespace(classifier="lr"),
        evaluation=SimpleNamespace(threshold=0.5),
        registry=SimpleNamespace(register_model=False),
    )
    load_splits = MagicMock(return_value=(MagicMock(), MagicMock()))
    training_result = MagicMock()
    setup_experiment = MagicMock(return_value="experiment-id")
    mlflow_run = MagicMock()

    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(train, "load_config", lambda **_: (config, Path("train.yaml")))
    monkeypatch.setattr(train, "load_and_validate_training_splits", load_splits)
    monkeypatch.setattr(train, "train_model", MagicMock(return_value=training_result))
    monkeypatch.setattr(train, "setup_local_experiment", setup_experiment)
    monkeypatch.setattr(train, "log_experiment_result", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock(return_value=mlflow_run))

    train.run_training_job(experiment_name=provided_name, split_name="baseline")

    assert setup_experiment.call_args.kwargs["experiment_name"] == expected_name
    load_splits.assert_called_once_with(
        split_name="baseline",
        split_dir=Path("data") / "splits",
    )


@pytest.mark.unit
def test_run_training_job_logs_tuning_result_and_fits_final_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(classifier="rf"),
        evaluation=SimpleNamespace(threshold=0.5),
        registry=SimpleNamespace(register_model=False),
        tuning=TuningConfig(
            enabled=True,
            n_trials=3,
            n_splits=2,
            time_column="reference_date",
            metric="roc_auc",
            random_state=19,
        ),
    )
    settings = SimpleNamespace(
        config_dir=Path("config"),
        data_dir=Path("data"),
        artifact_dir=Path("artifacts"),
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment_name="experiment",
    )
    train_df = MagicMock()
    test_df = MagicMock()
    load_splits = MagicMock(return_value=(train_df, test_df))
    optimizer = MagicMock(
        return_value=TuningResult(
            best_params={"n_estimators": 150, "max_depth": None},
            best_score=0.83,
            metric="roc_auc",
            n_trials=3,
            n_splits=2,
            time_column="reference_date",
            random_state=19,
        )
    )
    final_training = MagicMock()
    train_model = MagicMock(return_value=final_training)
    mlflow_run = MagicMock()
    log_params = MagicMock()
    log_metric = MagicMock()

    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(train, "load_config", lambda **_: (config, Path("train.yaml")))
    monkeypatch.setattr(train, "load_and_validate_training_splits", load_splits)
    monkeypatch.setattr(train, "optimize_hyperparameters", optimizer)
    monkeypatch.setattr(train, "train_model", train_model)
    monkeypatch.setattr(train, "setup_local_experiment", MagicMock(return_value="id"))
    monkeypatch.setattr(train, "log_experiment_result", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock(return_value=mlflow_run))
    monkeypatch.setattr(train.mlflow, "log_params", log_params)
    monkeypatch.setattr(train.mlflow, "log_metric", log_metric)

    result = train.run_training_job(split_name="temporal")

    assert result is final_training
    load_splits.assert_called_once_with(
        split_name="temporal",
        split_dir=Path("data") / "splits",
        preserve_time_column="reference_date",
    )
    optimizer.assert_called_once_with(config=config, train_df=train_df)
    train_model.assert_called_once_with(
        config=config,
        train_df=train_df,
        test_df=test_df,
        model_params_override={"n_estimators": 150, "max_depth": None},
        exclude_columns=["reference_date"],
    )
    log_params.assert_called_once_with(
        {
            "tuning_enabled": "true",
            "tuning_n_trials": 3,
            "tuning_n_splits": 2,
            "tuning_time_column": "reference_date",
            "tuning_metric": "roc_auc",
            "tuning_random_state": 19,
            "tuning_best_params": '{"max_depth": null, "n_estimators": 150}',
        }
    )
    log_metric.assert_called_once_with("tuning_cv_roc_auc", 0.83)
