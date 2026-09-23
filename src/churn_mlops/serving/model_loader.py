from dataclasses import dataclass

import mlflow
from sklearn.pipeline import Pipeline

from churn_mlops.config import ServingSettings
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

    # load settings used for model serving
    settings = ServingSettings()
    resolved_model_name = model_name or settings.model_name
    resolved_model_alias = model_alias or settings.model_alias
    resolved_tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    if resolved_tracking_uri:
        mlflow.set_tracking_uri(resolved_tracking_uri)

    # initialize model registry
    registry = ModelRegistry()

    # load model from registry
    model = registry.load_model(
        model_name=resolved_model_name,
        alias=resolved_model_alias,
    )

    # retrieve model version from registry
    version = registry.get_model_version_by_alias(
        model_name=resolved_model_name,
        alias=resolved_model_alias,
    )

    # retrieve class threshold from registry
    threshold = registry.get_threshold_by_alias(
        model_name=resolved_model_name,
        alias=resolved_model_alias,
    )

    # combine model metadata
    metadata = ModelMetadata(
        model_name=resolved_model_name,
        model_alias=resolved_model_alias,
        model_version=int(version.version),
        threshold=threshold,
    )

    return LoadedModel(
        model=model,
        metadata=metadata,
    )
