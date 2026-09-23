from pathlib import Path

import yaml

from churn_mlops.config.schemas import (
    ClassifierConfig,
    DataConfig,
    EvaluationConfig,
    FeatureBuilderConfig,
    PreprocessingConfig,
    RegistryConfig,
    TrainingConfig,
)
from churn_mlops.config.settings import RuntimeSettings


def load_config(
    file: str,
    path: Path | None = None,
) -> tuple[TrainingConfig, Path]:
    """Load a YAML training configuration into typed dataclasses.

    Args:
        file: Configuration filename.
        path: Directory containing the configuration file.

    Returns:
        A tuple containing the parsed ``TrainingConfig`` and the configuration
        file path used to load it.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If a required configuration section is missing.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    settings = RuntimeSettings()
    resolved_path = path or settings.config_dir

    config_file_path = resolved_path / file
    with open(config_file_path) as f:
        raw = yaml.safe_load(f)

    return TrainingConfig(
        data=DataConfig(**raw["data"]),
        feature_builder=FeatureBuilderConfig(**raw["feature_builder"]),
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        model=ClassifierConfig(**raw["model"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        registry=RegistryConfig(**raw["registry"]),
    ), config_file_path
