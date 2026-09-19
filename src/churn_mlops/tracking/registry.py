import mlflow
from mlflow import MlflowClient, MlflowException
from mlflow.entities.model_registry import ModelVersion
from sklearn.pipeline import Pipeline


class ModelRegistry:
    def __init__(self) -> None:
        self.client = MlflowClient()

    def register_model(self, model_uri: str, model_name: str) -> ModelVersion:
        """Register model from experiment run to ModelRegistry."""
        return mlflow.register_model(model_uri=model_uri, name=model_name)

    def set_alias(self, model_name: str, alias: str, version: int) -> None:
        """Set alias for a registered model version."""
        self.client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version,
        )

    def get_model_version(self, model_name: str, version: int) -> ModelVersion:
        """Retrieve model version from model name and version ID."""
        return self.client.get_model_version(
            name=model_name,
            version=str(version),
        )

    def get_model_version_by_alias(self, model_name: str, alias: str) -> ModelVersion:
        """Retrieve model version from model name and alias."""
        return self.client.get_model_version_by_alias(
            name=model_name,
            alias=alias,
        )

    def get_metric_by_alias(
        self, model_name: str, alias: str, metric_name: str
    ) -> float:
        """Retrieve specified metric from model based on alias."""
        version = self.get_model_version_by_alias(
            model_name=model_name,
            alias=alias,
        )
        run = self.client.get_run(version.run_id)
        return run.data.metrics[metric_name]

    def get_threshold_by_alias(self, model_name: str, alias: str) -> float:
        """Retrieve logged threshold from model based on alias."""
        version = self.get_model_version_by_alias(
            model_name=model_name,
            alias=alias,
        )
        run = self.client.get_run(version.run_id)
        return float(run.data.params["threshold"])

    def load_model(self, model_name: str, alias: str) -> Pipeline:
        """Load model from registry using model name and alias."""
        return mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")

    def get_champion_version(self, model_name: str) -> ModelVersion | None:
        try:
            return self.get_model_version_by_alias(
                model_name=model_name,
                alias="champion",
            )
        except MlflowException:
            return None
