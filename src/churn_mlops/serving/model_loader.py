from dataclasses import dataclass

import mlflow
from sklearn.pipeline import Pipeline

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
    model: Pipeline
    metadata: ModelMetadata


def load_model(
    tracking_uri: str | None = None,
    model_name: str | None = None,
    model_alias: str | None = None,
) -> LoadedModel:
    """Load the configured serving model and its registry metadata.

    Returns:
        Loaded model together with its name, alias, version, and probability
        threshold.

        Args:
            tracking_uri: Optional MLflow tracking URI.
            model_name: Optional registered model name.
            model_alias: Optional registered model alias.

    Raises:
        mlflow.exceptions.MlflowException: If the configured model or alias
            cannot be found in the MLflow registry.
    """

    # local tracking SQLite DB
    mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{TRACKING_DIR}/mlflow.db")

    # initialize model registry
    registry = ModelRegistry()

    # load settings used for model serving
    settings = ServingSettings()
    model_name = model_name or settings.model_name
    model_alias = model_alias or settings.model_alias

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
