from dataclasses import dataclass

import mlflow
from sklearn.base import BaseEstimator

from churn_mlops.config import ServingSettings
from churn_mlops.tracking import LoggedModelReference, ModelRegistry, TrainingManifest


@dataclass
class ModelMetadata:
    model_name: str
    model_alias: str | None
    model_version: int | None
    model_threshold: float
    model_reference: LoggedModelReference | None = None


@dataclass
class LoadedModel:
    model: BaseEstimator
    metadata: ModelMetadata


def load_model(
    tracking_uri: str | None = None,
    model_name: str | None = None,
    model_alias: str | None = None,
    model_version: int | None = None,
    manifest: TrainingManifest | None = None,
) -> LoadedModel:
    """Load a configured or manifest-identified model and its metadata.

    Args:
        tracking_uri: Optional MLflow tracking URI.
        model_name: Optional registered model name.
        model_alias: Optional registered model alias.
        model_version: Optional registered model version number.
        manifest: Optional training manifest identifying an exact model.

    Returns:
        Loaded model (trained estimator/pipeline) together with its
        model identity metadata and probability threshold.

    Raises:
        mlflow.exceptions.MlflowException: If the configured model or alias
            cannot be found in the MLflow registry.
        ValueError: If a manifest is combined with explicit model selectors.
    """
    if manifest is not None and any(
        selector is not None for selector in (model_name, model_alias, model_version)
    ):
        raise ValueError("A model manifest cannot be combined with model selectors.")

    settings = ServingSettings()
    tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = model_name or settings.model_name
    model_alias = model_alias or settings.model_alias
    model_version = model_version or settings.model_version
    model_reference = None
    if manifest is not None:
        model_name = manifest.model.model_name
        model_alias = None
        model_version = manifest.model.model_version
        model_threshold = manifest.training.threshold
        model_reference = manifest.model.model_reference
        if model_version is None:
            model = mlflow.sklearn.load_model(model_reference.model_uri)
        else:
            registry = ModelRegistry()
            model = registry.load_model_by_version(
                model_name=model_name,
                model_version=model_version,
            )
    else:
        registry = ModelRegistry()
        if model_version is not None:
            model_alias = None
        registered_model = registry.load_registered_model(
            model_name=model_name,
            model_alias=model_alias,
            model_version=model_version,
        )
        model = registered_model.model
        model_version = int(registered_model.model_version.version)
        model_threshold = registered_model.model_threshold

    return LoadedModel(
        model=model,
        metadata=ModelMetadata(
            model_name=model_name,
            model_alias=model_alias,
            model_version=model_version,
            model_threshold=model_threshold,
            model_reference=model_reference,
        ),
    )
