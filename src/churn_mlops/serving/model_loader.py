from dataclasses import dataclass

import mlflow

from churn_mlops.config import TRACKING_DIR, ServingSettings
from churn_mlops.tracking import ModelRegistry


@dataclass
class ModelMetadata:
    model_name: str
    model_alias: str
    model_version: int
    threshold: float


@dataclass
class LoadedModel:
    model: object
    metadata: ModelMetadata


def load_model() -> LoadedModel:

    # local tracking SQLite DB
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DIR}/mlflow.db")

    # initialize model registry
    registry = ModelRegistry()

    # load settings used for model serving
    settings = ServingSettings()
    model_name = settings.model_name
    model_alias = settings.model_alias

    # load model from registry
    model = registry.load_model(
        model_name=model_name,
        alias=model_alias,
    )

    # retrieve model version from registry
    version = registry.get_model_version_by_alias(
        model_name=model_name,
        alias=model_alias,
    )

    # retrieve class threshold from registry
    threshold = registry.get_threshold_by_alias(
        model_name=model_name,
        alias=model_alias,
    )

    # combine model metadata
    metadata = ModelMetadata(
        model_name=model_name,
        model_alias=model_alias,
        model_version=int(version.version),
        threshold=threshold,
    )

    return LoadedModel(
        model=model,
        metadata=metadata,
    )
