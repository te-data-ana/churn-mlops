from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from churn_mlops import train


@pytest.mark.unit
def test_train_main_passes_cli_config_to_training_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock the training boundary so this test covers CLI argument forwarding only.

    mock_run_training_job = MagicMock()

    monkeypatch.setattr(
        "churn_mlops.train.run_training_job",
        mock_run_training_job,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            "sample.yaml",
            "--experiment_name",
            "test_experiment",
        ],
    )

    train.main()

    mock_run_training_job.assert_called_once_with(
        config_file="sample.yaml", experiment_name="test_experiment"
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

    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(train, "load_config", lambda **_: (config, Path("train.yaml")))
    monkeypatch.setattr(train, "load_raw_data", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(train, "validate_training_data", lambda data: data)
    monkeypatch.setattr(train, "train_model", MagicMock(return_value=training_result))
    monkeypatch.setattr(train, "setup_local_experiment", setup_experiment)
    monkeypatch.setattr(train, "log_experiment_result", MagicMock())
    monkeypatch.setattr(train.mlflow, "get_experiment", MagicMock())
    monkeypatch.setattr(train.mlflow, "start_run", MagicMock(return_value=mlflow_run))

    train.run_training_job(experiment_name=provided_name)

    assert setup_experiment.call_args.kwargs["experiment_name"] == expected_name
