import logging

import mlflow
from mlflow import MlflowClient, MlflowException
from mlflow.entities.model_registry import ModelVersion
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        """Initialize a registry client using the active MLflow tracking URI."""
        self.client = MlflowClient()

    def register_model(self, model_uri: str, model_name: str) -> ModelVersion:
        """Register a logged model artifact under a registry name.

        Args:
            model_uri: MLflow URI identifying the logged model artifact.
            model_name: Name under which to register the model.

        Returns:
            The newly created MLflow model version.
        """
        try:
            logger.info("Registering model '%s' from URI '%s'.", model_name, model_uri)
            version = mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info(
                "Model '%s' registered as version %s.",
                model_name,
                version.version,
            )
            return version
        except Exception:
            logger.exception(
                "Failed to register model '%s' from '%s'.", model_name, model_uri
            )
            raise

    def set_alias(self, model_name: str, alias: str, version: int) -> None:
        """Assign an alias to a registered model version.

        Args:
            model_name: Registered model name.
            alias: Alias to assign.
            version: Model version receiving the alias.
        """
        try:
            logger.info(
                "Setting alias '%s' for model '%s' to version %s.",
                alias,
                model_name,
                version,
            )
            self.client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=version,
            )
        except Exception:
            logger.exception(
                "Failed to set alias '%s' for model '%s' to version %s.",
                alias,
                model_name,
                version,
            )
            raise

    def get_model_version(self, model_name: str, version: int) -> ModelVersion:
        """Retrieve a registered model version by name and version number.

        Args:
            model_name: Registered model name.
            version: Registered model version number.

        Returns:
            The matching MLflow model version.
        """
        return self.client.get_model_version(
            name=model_name,
            version=str(version),
        )

    def get_model_version_by_alias(self, model_name: str, alias: str) -> ModelVersion:
        """Retrieve a registered model version by model name and alias.

        Args:
            model_name: Registered model name.
            alias: Alias identifying the desired version.

        Returns:
            The model version currently assigned to the alias.
        """
        try:
            logger.info(
                "Looking up model version for model '%s' with alias '%s'.",
                model_name,
                alias,
            )
            return self.client.get_model_version_by_alias(
                name=model_name,
                alias=alias,
            )
        except MlflowException:
            logger.warning(
                "No model version found for '%s' with alias '%s'.", model_name, alias
            )
            raise

    def get_metric_by_alias(
        self, model_name: str, alias: str, metric_name: str
    ) -> float:
        """Retrieve a metric from the run associated with an alias.

        Args:
            model_name: Registered model name.
            alias: Alias identifying the model version and run.
            metric_name: Name of the logged metric.

        Returns:
            Logged metric value.
        """
        version = self.get_model_version_by_alias(
            model_name=model_name,
            alias=alias,
        )
        run = self.client.get_run(version.run_id)
        logger.info(
            "Fetched metric '%s' for model '%s' alias '%s'.",
            metric_name,
            model_name,
            alias,
        )
        return run.data.metrics[metric_name]

    def get_threshold_by_alias(self, model_name: str, alias: str) -> float:
        """Retrieve the logged classification threshold for an alias.

        Args:
            model_name: Registered model name.
            alias: Alias identifying the model version and run.

        Returns:
            Logged threshold converted to ``float``.
        """
        version = self.get_model_version_by_alias(
            model_name=model_name,
            alias=alias,
        )
        run = self.client.get_run(version.run_id)
        threshold = float(run.data.params["threshold"])
        logger.info(
            "Fetched threshold %.4f for model '%s' alias '%s'.",
            threshold,
            model_name,
            alias,
        )
        return threshold

    def load_model(self, model_name: str, alias: str) -> Pipeline:
        """Load a scikit-learn model from the MLflow registry.

        Args:
            model_name: Registered model name.
            alias: Alias identifying the version to load.

        Returns:
            Loaded scikit-learn pipeline.
        """
        try:
            logger.info("Loading model '%s' with alias '%s'.", model_name, alias)
            model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
            logger.info(
                "Model '%s' with alias '%s' loaded successfully.", model_name, alias
            )
            return model
        except Exception:
            logger.exception(
                "Failed to load model '%s' with alias '%s'.", model_name, alias
            )
            raise

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
                alias="champion",
            )
            logger.info(
                "Champion model for '%s' is version %s.", model_name, champion.version
            )
            return champion
        except MlflowException:
            logger.warning("No champion alias exists for model '%s'.", model_name)
            return None
