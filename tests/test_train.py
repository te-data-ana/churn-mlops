from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock

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
    config_file = "train.yaml"
    split_name = "temporal"
    load_splits = MagicMock(return_value=(MagicMock(), MagicMock()))
    train_model_mock = MagicMock(return_value=MagicMock())
    setup_experiment = MagicMock(return_value="experiment-id")
    mlflow_run = MagicMock()
    get_active_run = train.mlflow.active_run
    assert get_active_run() is None

    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(train, "load_config", lambda **_: (config, Path(config_file)))
    monkeypatch.setattr(train, "load_and_validate_training_splits", load_splits)
    monkeypatch.setattr(train, "train_model", train_model_mock)
    monkeypatch.setattr(train, "setup_local_experiment", setup_experiment)
    monkeypatch.setattr(train, "log_experiment_result", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock(return_value=mlflow_run))
    monkeypatch.setattr(
        train.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-id")),
    )
    monkeypatch.setattr(train.mlflow, "set_tags", MagicMock())
    set_tag = MagicMock()
    monkeypatch.setattr(train.mlflow, "set_tag", set_tag)

    train.run_training_job(
        config_file=config_file,
        split_name=split_name,
        exclude_columns=["customerid"],
        experiment_name=provided_name,
        register_model=False,
        write_manifest=False,
    )

    assert setup_experiment.call_args.kwargs["experiment_name"] == expected_name
    assert get_active_run() is None
    set_tag.assert_any_call("git_commit", ANY)
    load_splits.assert_called_once_with(
        split_name=split_name,
        split_dir=Path("data") / "splits",
    )
    train_model_mock.assert_called_once_with(
        config=config,
        train_df=load_splits.return_value[0],
        test_df=load_splits.return_value[1],
        exclude_columns=["customerid"],
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
    config_file = "train.yaml"
    split_name = "temporal"
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
    monkeypatch.setattr(train, "load_config", lambda **_: (config, Path(config_file)))
    monkeypatch.setattr(train, "load_and_validate_training_splits", load_splits)
    monkeypatch.setattr(train, "optimize_hyperparameters", optimizer)
    monkeypatch.setattr(train, "train_model", train_model)
    monkeypatch.setattr(train, "setup_local_experiment", MagicMock(return_value="id"))
    monkeypatch.setattr(train, "log_experiment_result", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock(return_value=mlflow_run))
    monkeypatch.setattr(
        train.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-id")),
    )
    monkeypatch.setattr(train.mlflow, "set_tags", MagicMock())
    monkeypatch.setattr(train.mlflow, "set_tag", MagicMock())
    monkeypatch.setattr(train.mlflow, "log_params", log_params)
    monkeypatch.setattr(train.mlflow, "log_metric", log_metric)

    result = train.run_training_job(
        config_file=config_file,
        split_name=split_name,
        exclude_columns=["customerid"],
        register_model=False,
        write_manifest=False,
    )

    assert result is final_training
    load_splits.assert_called_once_with(
        split_name=split_name,
        split_dir=Path("data") / "splits",
        preserve_time_column="reference_date",
    )
    optimizer.assert_called_once_with(
        config=config,
        train_df=train_df,
        exclude_columns=["customerid"],
    )
    train_model.assert_called_once_with(
        config=config,
        train_df=train_df,
        test_df=test_df,
        model_params_override={"n_estimators": 150, "max_depth": None},
        exclude_columns=["customerid", "reference_date"],
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


@pytest.mark.unit
def test_experiment_registration_does_not_update_aliases_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        data=SimpleNamespace(target_column="churn"),
        model=SimpleNamespace(classifier="rf"),
        evaluation=SimpleNamespace(threshold=0.5),
        registry=SimpleNamespace(
            register_model=False,
            promote_model=True,
            registry_params={"model_name": "churn-propensity", "alias": "candidate"},
        ),
        tuning=TuningConfig(),
    )
    settings = SimpleNamespace(
        config_dir=tmp_path,
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment_name="experiment",
    )
    training_result = SimpleNamespace(
        metrics={"roc_auc": 0.84},
        classifier_config={"model_name": "churn-propensity"},
        metadata={
            "feature_count": 2,
            "feature_names_in": ["feature_in_1", "feature_in_2"],
            "feature_names_out": ["feature_out_1", "feature_out_2"],
        },
    )
    registered_version = SimpleNamespace(version=7, run_id="training-run")
    registry = MagicMock()
    registry.register_model.return_value = registered_version
    model_card_builder = MagicMock()
    model_card_builder.build.return_value = "# Model Card"

    config_file = "train.yaml"
    split_name = "temporal"
    classifier_alias = config.model.classifier
    run_name = f"{classifier_alias}_{split_name}"
    manifest_file = settings.data_dir / f"manifests/{run_name}.json"

    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(
        train, "load_config", lambda **_: (config, tmp_path / config_file)
    )
    monkeypatch.setattr(
        train, "load_and_validate_training_splits", MagicMock(return_value=(None, None))
    )
    monkeypatch.setattr(train, "train_model", MagicMock(return_value=training_result))
    monkeypatch.setattr(train, "setup_local_experiment", MagicMock(return_value="exp"))
    monkeypatch.setattr(
        train,
        "log_experiment_result",
        MagicMock(return_value=SimpleNamespace(model_uri="runs:/training-run/model")),
    )
    monkeypatch.setattr(train, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(train, "ModelCardBuilder", lambda: model_card_builder)
    monkeypatch.setattr(train, "log_model_card", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_tracking_uri", MagicMock(return_value="uri"))
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock())
    monkeypatch.setattr(train.mlflow, "log_artifact", MagicMock())
    monkeypatch.setattr(
        train.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="training-run")),
    )
    set_tags = MagicMock()
    monkeypatch.setattr(train.mlflow, "set_tags", set_tags)
    set_tag = MagicMock()
    monkeypatch.setattr(train.mlflow, "set_tag", set_tag)

    result = train.run_training_job(
        config_file=config_file,
        split_name=split_name,
        register_model=True,
        promote_model=False,
        write_manifest=True,
    )

    assert result is training_result
    registry.register_model.assert_called_once_with(
        model_uri="runs:/training-run/model",
        model_name="churn-propensity",
    )
    registry.set_alias.assert_not_called()
    registry.get_champion_version.assert_not_called()
    assert manifest_file.is_file()
    from churn_mlops.tracking import TrainingManifest

    manifest = TrainingManifest.read(manifest_file)
    assert manifest.model_name == "churn-propensity"
    assert manifest.model_version == 7
    assert manifest.run_id == "training-run"
    assert manifest.run_name == run_name
    set_tag.assert_any_call("git_commit", ANY)
    train.mlflow.log_artifact.assert_called_once_with(
        local_path=str(manifest_file),
        artifact_path="manifest",
    )
