from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models.model import ModelInfo

from churn_mlops.config.settings import RuntimeSettings

if TYPE_CHECKING:
    from churn_mlops.config.schemas import TrainingConfig
    from churn_mlops.training import TrainingResult


def setup_local_experiment(
    experiment_name: str,
    tracking_uri: str | None = None,
    artifact_dir: Path | None = None,
) -> str:
    """Configure local MLflow tracking and select an experiment.

    Args:
        experiment_name: Name of the experiment to create or activate.
        tracking_uri: Optional MLflow tracking URI. Defaults to the local
            project database.
        artifact_dir: Directory used for experiment artifacts.

    Returns:
        MLflow experiment identifier.

    Raises:
        MlflowException: If the experiment cannot be created or retrieved.
    """

    settings = RuntimeSettings()
    resolved_tracking_uri = tracking_uri or settings.mlflow_tracking_uri
    resolved_artifact_dir = artifact_dir or settings.artifact_dir
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(resolved_tracking_uri)

    # create experiment in case it does not exist yet under that name
    try:
        experiment_id = mlflow.create_experiment(
            experiment_name, artifact_location=str(resolved_artifact_dir)
        )
    # otherwise retrieve corresponding ID of existing experiment
    except MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # set experiment using its ID
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def log_experiment_result(
    result: TrainingResult,
    config: TrainingConfig,
    config_file_path: Path,
    artifact_dir: Path | None = None,
) -> ModelInfo:
    """Log training parameters, metrics, artifacts, and the fitted pipeline.

    Args:
        result: Training result containing the fitted pipeline, metrics,
            classifier configuration, and feature metadata.
        config: Training configuration whose settings are logged as parameters.
        config_file_path: Path to the full configuration file artifact.
        artifact_dir: Directory used for supporting artifacts.

    Returns:
        MLflow model metadata for the logged scikit-learn model.
    """

    resolved_artifact_dir = artifact_dir or RuntimeSettings().artifact_dir

    # log all parameters required for reproducibility
    # log data parameters
    mlflow.log_params(
        {
            "target_column": config.data.target_column,
            "test_size": config.data.test_size,
            "data_random_state": config.data.random_state,
        }
    )
    # log feature enginnering hyper-parameters
    mlflow.log_params(config.feature_builder.feature_params)
    # log preprocessing hyper-parameters
    mlflow.log_params(
        {
            "numeric_imputer": config.preprocessing.numeric_impute_strategy,
            "categorical_imputer": config.preprocessing.categorical_impute_strategy,
        }
    )
    # log classifier and its hyper-parameters
    mlflow.log_params(result.classifier_config)
    # log model evaluation parameters required for reproducibility
    mlflow.log_param("threshold", config.evaluation.threshold)

    # log additional parameters on training data
    mlflow.log_params(
        {
            "train_rows": result.metadata["train_rows"],
            "test_rows": result.metadata["test_rows"],
            "feature_count": result.metadata["feature_count"],
        }
    )
    # log a relevant tags
    mlflow.set_tags(
        {
            "pipeline_steps": list(result.trained_pipeline.named_steps.keys()),
            # "git_branch": current_branch,
            # "git_commit": commit_hash,
        }
    )

    # log all model metrics from churn_mlops.evaluation import evaluate_model
    mlflow.log_metrics(result.metrics)

    # log trained model pipeline artifact
    model_info = mlflow.sklearn.log_model(
        result.trained_pipeline,
        name="model",
        skops_trusted_types=[
            "churn_mlops.models.features.FeatureBuilder",
            "numpy.dtype",
            "numpy.number",
            "sklearn.compose._column_transformer.make_column_selector",
        ],
    )

    # log model feature names
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_names_path = resolved_artifact_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(
            {
                "input": result.metadata["feature_names_in"],
                "output": result.metadata["feature_names_out"],
            },
            f,
        )
    mlflow.log_artifact(feature_names_path, artifact_path="features")

    # log full config file as artifact
    mlflow.log_artifact(config_file_path, artifact_path="config")

    return model_info
