from unittest.mock import MagicMock

import pytest

from churn_mlops import train


@pytest.mark.unit
def test_main(monkeypatch):

    mock_run_training_job = MagicMock()

    monkeypatch.setattr(
        "churn_mlops.train.run_training_job",
        mock_run_training_job,
    )

    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--config", "sample_training_config.yaml"],
    )

    train.main()

    mock_run_training_job.assert_called_once_with(
        config_file="sample_training_config.yaml"
    )
