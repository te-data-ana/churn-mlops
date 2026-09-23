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
