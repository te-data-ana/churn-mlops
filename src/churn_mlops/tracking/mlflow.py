from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models.model import ModelInfo

from churn_mlops.config.settings import RuntimeSettings

logger = logging.getLogger(__name__)

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

    logger.info("Logging training parameters and tags to MLflow.")
    # log all parameters required for reproducibility
    # log classifier and its hyper-parameters
    mlflow.log_params(result.classifier_config)
    # log data parameters
    mlflow.log_params(
        {
            "column_target": config.data.target_column,
            "column_time": config.data.time_column,
        }
    )
    # log feature enginnering hyper-parameters
    mlflow.log_params(config.feature_builder.feature_params)
    # log preprocessing hyper-parameters
    mlflow.log_params(
        {
            "imputer_numeric": config.preprocessing.numeric_impute_strategy,
            "imputer_categorical": config.preprocessing.categorical_impute_strategy,
        }
    )
    # log calibration parameters
    calibration_parameters: dict[str, str | int] = {
        "calibration_method": config.calibration.method,
        "calibration_n_splits": config.calibration.n_splits,
        "calibration_n_bins": config.calibration.n_bins,
    }
    mlflow.log_params(calibration_parameters)
    # log threshold parameters
    threshold_parameters: dict[str, str | float | int] = {
        "threshold": result.threshold,
        "threshold_strategy": config.threshold.strategy,
        "threshold_data": result.threshold_data,
    }
    mlflow.log_params(threshold_parameters)
    # log promotion parameters
    if config.promotion.promote_model:
        promotion_parameters = {
            f"promotion_weight_{metric}": weight
            for metric, weight in config.promotion.metric_weights.items()
        }
        promotion_parameters["promotion_min_delta"] = config.promotion.min_delta
        mlflow.log_params(promotion_parameters)

    # log relevant tags
    mlflow.set_tags(
        {
            "calibration": config.calibration.method,
            "threshold_data": result.threshold_data,
            "feature_count": len(result.feature_names_out),
        }
    )
    # mlflow.log_param("feature_count", len(result.feature_names_out))

    logger.info("Logging evaluation metrics to MLflow: %s.", result.metrics)
    mlflow.log_metrics(result.metrics)
    logger.info("Logging calibration reliability diagram to MLflow.")
    mlflow.log_figure(
        result.calibration_figure,
        artifact_file="calibration/reliability_diagram.png",
    )

    logger.info("Logging fitted model pipeline to MLflow.")
    model_info = mlflow.sklearn.log_model(
        result.trained_pipeline,
        name="model",
        skops_trusted_types=[
            "churn_mlops.models.features.FeatureBuilder",
            "numpy.dtype",
            "numpy.number",
            "sklearn.compose._column_transformer.make_column_selector",
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration._SigmoidCalibration",
            "sklearn.isotonic.IsotonicRegression",
            "sklearn.tree._tree.Tree",
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
        ],
    )
    logger.info("Logged fitted model pipeline at '%s'.", model_info.model_uri)

    logger.info("Logging training configuration artifact '%s'.", config_file_path)
    mlflow.log_artifact(config_file_path, artifact_path="config")

    return model_info
