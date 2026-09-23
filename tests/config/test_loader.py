import pytest

from churn_mlops.config import load_config


@pytest.mark.unit
def test_load_config_returns_config_and_resolved_path(sample_config_yaml) -> None:
    config_file_name = sample_config_yaml.name
    config_file_dir = sample_config_yaml.parent

    config, path = load_config(
        file=config_file_name,
        path=config_file_dir,
    )

    assert config.data.target_column == "churn"
    assert path == sample_config_yaml
