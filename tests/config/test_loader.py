import pytest

from churn_mlops.config import CONFIG_DIR, load_config


@pytest.mark.unit
def test_load_config():
    config_file_name = "sample_training_config.yaml"
    config_file_dir = CONFIG_DIR
    config_file = config_file_dir / config_file_name

    config, path = load_config(
        file=config_file_name,
        path=config_file_dir,
    )

    assert config.data.target_column == "churn"
    assert path == config_file
