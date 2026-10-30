from __future__ import annotations

import json
import tempfile
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
    # log classifier and its hyper-parameters
    mlflow.log_params(result.classifier_config)
    # log data parameters
    mlflow.log_params(
        {
            "target_column": config.data.target_column,
            "time_column": config.data.time_column,
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
        }
    )
    mlflow.log_param("feature_count", len(result.feature_names_out))

    # log all model metrics
    mlflow.log_metrics(result.metrics)
    mlflow.log_figure(
        result.calibration_figure,
        artifact_file="calibration/reliability_diagram.png",
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        threshold_path = Path(temporary_directory) / "decision_threshold.json"
        threshold_path.write_text(
            json.dumps(
                {
                    "calibration_method": config.calibration.method,
                    "threshold_strategy": config.threshold.strategy,
                    "threshold_data": result.threshold_data,
                    "threshold": result.threshold,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(
            threshold_path,
            artifact_path="evaluation",
        )

    # log trained model pipeline artifact
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

    # log model feature names
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    feature_names_path = resolved_artifact_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(
            {
                "input": result.feature_names_in,
                "output": result.feature_names_out,
            },
            f,
        )
    mlflow.log_artifact(feature_names_path, artifact_path="features")

    # log full config file as artifact
    mlflow.log_artifact(config_file_path, artifact_path="config")

    return model_info
