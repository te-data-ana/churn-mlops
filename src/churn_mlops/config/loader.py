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
from churn_mlops.config.settings import CONFIG_DIR


def load_config(
    file: str = "sample_training_config.yaml", path: str = CONFIG_DIR
) -> TrainingConfig:
    config_file_path = path / file
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
