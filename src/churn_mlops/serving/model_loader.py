from dataclasses import dataclass

import mlflow
from sklearn.base import BaseEstimator

from churn_mlops.config import ServingSettings
from churn_mlops.tracking import ModelRegistry


@dataclass
class ModelMetadata:
    model_name: str
    model_alias: str | None
    model_version: int | None
    threshold: float
    model_uri: str | None = None


@dataclass
class LoadedModel:
    model: BaseEstimator
    metadata: ModelMetadata


def load_model(
    tracking_uri: str | None = None,
    model_name: str | None = None,
    model_alias: str | None = None,
    model_version: int | None = None,
    model_uri: str | None = None,
    model_threshold: float | None = None,
) -> LoadedModel:
    """Load the configured serving model and its registry metadata.

    Args:
        tracking_uri: Optional MLflow tracking URI.
        model_name: Optional registered model name.
        model_alias: Optional registered model alias.
        model_version: Optional immutable registered model version. When
            supplied, it takes precedence over the alias.
        model_uri: Optional logged MLflow model URI. When supplied, it
            bypasses the model registry.
        model_threshold: Threshold accompanying a logged model URI.

    Returns:
        Loaded model together with its name, alias, version, and
        probability threshold.

    Raises:
        mlflow.exceptions.MlflowException: If the configured model or alias
            cannot be found in the MLflow registry.
        ValueError: If URI-based loading has invalid or incomplete metadata.
    """

    if model_uri is None and model_threshold is not None:
        raise ValueError("A model threshold can only be used with a model URI.")

    # load settings used for model serving
    settings = ServingSettings()
    resolved_model_name = model_name or settings.model_name
    resolved_model_alias = model_alias or settings.model_alias
    resolved_tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    if resolved_tracking_uri:
        mlflow.set_tracking_uri(resolved_tracking_uri)

    if model_uri is not None:
        if not model_uri:
            raise ValueError("A model URI cannot be empty.")
        if model_version is not None:
            raise ValueError("A model URI cannot be combined with a model version.")
        if model_threshold is None:
            raise ValueError("A model URI requires its recorded prediction threshold.")
        if not 0 <= model_threshold <= 1:
            raise ValueError("A model threshold must be between 0 and 1.")
        model = mlflow.sklearn.load_model(model_uri)
        version_number = None
        threshold = model_threshold
        resolved_model_alias = None
    elif model_version is not None:
        registry = ModelRegistry()
        model = registry.load_model_by_version(
            model_name=resolved_model_name,
            version=model_version,
        )
        version_number = model_version
        threshold = registry.get_threshold_by_version(
            model_name=resolved_model_name,
            version=model_version,
        )
        resolved_model_alias = None
    else:
        registry = ModelRegistry()
        model = registry.load_model_by_alias(
            model_name=resolved_model_name,
            alias=resolved_model_alias,
        )
        version = registry.get_model_version_by_alias(
            model_name=resolved_model_name,
            alias=resolved_model_alias,
        )
        version_number = int(version.version)
        threshold = registry.get_threshold_by_alias(
            model_name=resolved_model_name,
            alias=resolved_model_alias,
        )

    # combine model metadata
    metadata = ModelMetadata(
        model_name=resolved_model_name,
        model_alias=resolved_model_alias,
        model_version=version_number,
        threshold=threshold,
        model_uri=model_uri,
    )

    return LoadedModel(
        model=model,
        metadata=metadata,
    )
