from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from churn_mlops import train


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
    config = SimpleNamespace(
        model=SimpleNamespace(classifier="lr"),
        evaluation=SimpleNamespace(threshold=0.5),
        registry=SimpleNamespace(register_model=False),
    )
    settings = SimpleNamespace(
        config_dir=Path("config"),
        data_dir=Path("data"),
        artifact_dir=Path("artifacts"),
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment_name=configured_name,
    )
    training_result = MagicMock()
    setup_experiment = MagicMock(return_value="experiment-id")
    mlflow_run = MagicMock()

    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(train, "load_config", lambda **_: (config, Path("train.yaml")))
    load_splits = MagicMock(return_value=(MagicMock(), MagicMock()))
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
