import logging
from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient, MlflowException
from mlflow.entities.model_registry import ModelVersion
from mlflow.models.model import ModelInfo
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredModel:
    """Loaded registry artifact (model/version) along with class threshold."""

    model: BaseEstimator
    model_version: ModelVersion
    model_threshold: float


class ModelRegistry:
    def __init__(self) -> None:
        """Initialize a registry client using the active MLflow tracking URI."""
        self.client = MlflowClient()

    def register_model(self, model_info: ModelInfo, model_name: str) -> ModelVersion:
        """Register a logged model artifact under a registry name.

        Args:
            model_info: Metadata for the logged MLflow model artifact.
            model_name: Name under which to register the model.

        Returns:
            The newly created MLflow model version.
        """
        try:
            logger.info(
                "Registering model '%s' from URI '%s'.",
                model_name,
                model_info.model_uri,
            )
            version = mlflow.register_model(
                model_uri=model_info.model_uri,
                name=model_name,
            )
            logger.info(
                "Model '%s' registered as version %s.",
                model_name,
                version.version,
            )
            return version
        except Exception:
            logger.exception(
                "Failed to register model '%s' from '%s'.",
                model_name,
                model_info.model_uri,
            )
            raise

    def set_alias(self, model_name: str, model_alias: str, model_version: int) -> None:
        """Assign an alias to a registered model version.

        Args:
            model_name: Registered model name.
            model_alias: Alias to assign.
            model_version: Model version receiving the alias.
        """
        try:
            logger.info(
                "Setting alias '%s' for model '%s' to version %s.",
                model_alias,
                model_name,
                model_version,
            )
            self.client.set_registered_model_alias(
                name=model_name,
                alias=model_alias,
                version=model_version,
            )
        except Exception:
            logger.exception(
                "Failed to set alias '%s' for model '%s' to version %s.",
                model_alias,
                model_name,
                model_version,
            )
            raise

    def get_model_version(self, model_name: str, model_version: int) -> ModelVersion:
        """Retrieve a registered model version by name and version number.

        Args:
            model_name: Registered model name.
            model_version: Registered model version number.

        Returns:
            The matching MLflow model version.
        """
        return self.client.get_model_version(
            name=model_name,
            version=str(model_version),
        )

    def get_model_version_by_alias(
        self, model_name: str, model_alias: str
    ) -> ModelVersion:
        """Retrieve a registered model version by model name and alias.

        Args:
            model_name: Registered model name.
            model_alias: Alias identifying the desired version.

        Returns:
            The model version currently assigned to the alias.
        """
        try:
            logger.info(
                "Looking up model version for model '%s' with alias '%s'.",
                model_name,
                model_alias,
            )
            return self.client.get_model_version_by_alias(
                name=model_name,
                alias=model_alias,
            )
        except MlflowException:
            logger.warning(
                "No model version found for '%s' with alias '%s'.",
                model_name,
                model_alias,
            )
            raise

    def update_model_description(
        self, model_name: str, model_version: int, description: str
    ) -> ModelVersion:
        """Update the description of a registered model version.

        Args:
            model_name: Registered model name.
            model_version: Registered model version number.
            description: New description text.

        Returns:
            The MLflow model version with updated description.
        """
        return self.client.update_model_version(
            name=model_name, version=str(model_version), description=description
        )

    def get_metric_by_alias(
        self, model_name: str, model_alias: str, metric_name: str
    ) -> float:
        """Retrieve a metric from the run associated with an alias.

        Args:
            model_name: Registered model name.
            model_alias: Alias identifying the model version and run.
            metric_name: Name of the logged metric.

        Returns:
            Logged metric value.
        """
        version = self.get_model_version_by_alias(
            model_name=model_name,
            model_alias=model_alias,
        )
        run = self.client.get_run(run_id=version.run_id)
        logger.info(
            "Fetched metric '%s' for model '%s' alias '%s'.",
            metric_name,
            model_name,
            model_alias,
        )
        return run.data.metrics[metric_name]

    def get_metrics_by_alias(
        self, model_name: str, model_alias: str
    ) -> dict[str, float]:
        """Retrieve all logged metrics for the run associated with an alias.

        Args:
            model_name: Registered model name.
            alias: Alias identifying the model version and run.

        Returns:
            The metric names and values logged by the model's training run.
        """
        version = self.get_model_version_by_alias(
            model_name=model_name,
            model_alias=model_alias,
        )
        run = self.client.get_run(run_id=version.run_id)
        return {name: float(value) for name, value in run.data.metrics.items()}

    def load_model_by_version(
        self, model_name: str, model_version: int
    ) -> BaseEstimator:
        """Load a trained model pipeline from an immutable registry version.

        Args:
            model_name: Registered model name.
            model_version: Exact registered version number.

        Returns:
            Trained model pipeline from the registry.
        """
        try:
            logger.info("Loading model '%s' version %s.", model_name, model_version)
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
            logger.info(
                "Model '%s' version %s loaded successfully.",
                model_name,
                model_version,
            )
            return model
        except Exception:
            logger.exception(
                "Failed to load model '%s' version %s.",
                model_name,
                model_version,
            )
            raise

    def load_registered_model(
        self,
        model_name: str,
        model_alias: str | None = None,
        model_version: int | None = None,
    ) -> RegisteredModel:
        """Load a registry model and threshold with one version/run resolution.

        Args:
            model_name: Registered model name.
            model_alias: Alias identifying the desired version.
            model_version: Exact registered version number.

        Returns:
            Loaded estimator, its registry model version, and class threshold.

        Raises:
            ValueError: If exactly one of alias or version is not supplied.
        """
        if (model_alias is None) == (model_version is None):
            raise ValueError("Provide exactly one model alias or version.")

        model_version_object = (
            self.get_model_version_by_alias(
                model_name=model_name, model_alias=model_alias
            )
            if model_alias is not None
            else self.get_model_version(
                model_name=model_name, model_version=model_version
            )
        )
        resolved_version = int(model_version_object.version)
        model = self.load_model_by_version(
            model_name=model_name,
            model_version=resolved_version,
        )
        run = self.client.get_run(run_id=model_version_object.run_id)
        threshold = float(run.data.params["threshold"])
        logger.info(
            "Loaded model '%s' version %s with threshold %.4f.",
            model_name,
            resolved_version,
            threshold,
        )
        return RegisteredModel(
            model=model,
            model_version=model_version_object,
            model_threshold=threshold,
        )

    def get_champion_version(self, model_name: str) -> ModelVersion | None:
        """Return the model version assigned to ``champion`` if available.

        Args:
            model_name: Registered model name.

        Returns:
            Champion model version, or ``None`` when the alias is unavailable.
        """
        try:
            champion = self.get_model_version_by_alias(
                model_name=model_name,
                model_alias="champion",
            )
            logger.info(
                "Champion model for '%s' is version %s.", model_name, champion.version
            )
            return champion
        except MlflowException:
            logger.warning("No champion alias exists for model '%s'.", model_name)
            return None
