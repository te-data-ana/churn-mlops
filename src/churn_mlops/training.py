import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_mlops.config import (
    ARTIFACT_DIR,
    CONFIG_DIR,
    RAW_DATA_DIR,
    configure_logging,
    load_config,
)
from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.data import load_raw_data, validate_training_data
from churn_mlops.evaluation import Timer, evaluate_model
from churn_mlops.models import build_classifier_pipeline, create_model
from churn_mlops.tracking import (
    ModelRegistry,
    PromotionService,
    log_experiment_result,
    setup_local_experiment,
)


@dataclass
class TrainingResult:
    trained_pipeline: Pipeline
    metrics: dict[str, float]
    classifier_config: dict[str, Any]
    metadata: dict[str, Any]


def run_training_job(
    config_file: str = "sample_training_config.yaml",
    config_dir: Path = CONFIG_DIR,
    training_file: str = "customer_churn_dataset-training.csv",
    index_col: str | None = "customerid",
    data_dir: Path = RAW_DATA_DIR,
    experiment_name: str = "test",
    tracking_uri: str | None = None,
    artifact_dir: Path = ARTIFACT_DIR,
) -> TrainingResult:
    """Run the configured training, tracking, registration, and promotion flow.

    Args:
        config_file: Training configuration YAML filename.
        config_dir: Directory containing the training configuration.
        training_file: Raw training data CSV filename.
        index_col: Column to use as the training DataFrame index.
        data_dir: Directory containing the training CSV.
        experiment_name: Local MLflow experiment name.
        tracking_uri: Optional MLflow tracking URI.
        artifact_dir: Directory used for training artifacts.

    Returns:
        TrainingResult containing the fitted pipeline, metrics, classifier
        configuration, and training metadata.

    Raises:
        FileNotFoundError: If the configuration or training data file is not
            found.
        pandera.errors.SchemaError: If the training data violates its schema.
        ValueError: If the configured classifier or promotion operation is
            invalid.

    Side Effects:
        Logs an MLflow run and, when enabled by configuration, registers the
        candidate model and updates registry aliases.
    """

    # Load and establish all required configurations
    configure_logging()
    logger = logging.getLogger(__name__)

    try:
        config, config_file_path = load_config(file=config_file, path=config_dir)
        classifier_alias = config.model.classifier
        eval_threshold = config.evaluation.threshold

        logger.info("Loaded training configuration from '%s'.", config_file_path)
        logger.info(
            "Configured model alias '%s' with evaluation threshold %.4f.",
            classifier_alias,
            eval_threshold,
        )

        run_name = f"{classifier_alias}_{eval_threshold!s}"
        experiment_id = setup_local_experiment(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            artifact_dir=artifact_dir,
        )
        experiment = mlflow.get_experiment(experiment_id)
        logger.info(
            "MLflow experiment '%s' initialized with id '%s'.",
            experiment_name,
            experiment_id,
        )
        logger.info("Tracking URI: %s", mlflow.get_tracking_uri())
        logger.info("Experiment metadata: %s", experiment)

        # Load and validate the training data
        df = load_raw_data(
            file_name=training_file, index_col=index_col, data_dir=data_dir
        )
        df = validate_training_data(df)

        # Start an MLflow run for the training job and log the results
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            result = train(config, df)
            model_info = log_experiment_result(
                result=result,
                config=config,
                config_file_path=config_file_path,
                artifact_dir=artifact_dir,
            )

        # Register the candidate model and evaluate it for promotion to champion if enabled
        if config.registry.register_model:
            model_name = config.registry.registry_params["model_name"]
            model_alias = config.registry.registry_params["alias"]

            logger.info(
                "Starting registration of '%s' model with alias '%s':",
                model_name,
                model_alias,
            )
            registry = ModelRegistry()

            candidate = registry.register_model(
                model_uri=model_info.model_uri,
                model_name=model_name,
            )
            registry.set_alias(
                model_name=model_name,
                alias=model_alias,
                version=candidate.version,
            )

            champion = registry.get_champion_version(model_name=model_name)

            candidate_metric = result.metrics["roc_auc"]
            if champion:
                logger.info("Retrieving metric for current champion model.")
                champion_metric = registry.get_metric_by_alias(
                    model_name=model_name,
                    alias="champion",
                    metric_name="roc_auc",
                )
            else:
                logger.info(
                    "No champion model exists in registry for '%s'.",
                    model_name,
                )
                champion_metric = None

            # Promotion requires a strict improvement over the configured delta.
            promotion_delta = config.registry.registry_params["promotion_delta"]
            promotion_service = PromotionService()
            promotion_decision = promotion_service.evaluate_candidate(
                candidate_metric=candidate_metric,
                champion_metric=champion_metric,
                promotion_delta=promotion_delta,
            )

            if promotion_decision.promote:
                logger.info("Decision to promote candidate model to champion:")
                logger.info(promotion_decision.reason)
                if promotion_decision.metric_delta:
                    logger.info(
                        "The AUC delta was '%.4f'.", promotion_decision.metric_delta
                    )
                promotion_service.promote_candidate(
                    decision=promotion_decision,
                    registry=registry,
                    model_name=model_name,
                    candidate_version=candidate.version,
                )
                logger.info(
                    "Candidate model '%s' version %s was promoted to champion.",
                    model_name,
                    candidate.version,
                )
            else:
                logger.warning(
                    "No promotion of candidate model: %s", promotion_decision.reason
                )

        return result
    except Exception:
        logger.exception(
            "Training job failed while processing config '%s' and file '%s'.",
            config_file,
            training_file,
        )
        raise


def train(config: TrainingConfig, df: pd.DataFrame) -> TrainingResult:
    """Fit and evaluate a configured classifier pipeline.

    Args:
        config: Training, feature, preprocessing, model, evaluation and
            MLflow registration settings.
        df: Validated DataFrame containing features and the configured target.

    Returns:
        TrainingResult containing the fitted pipeline, evaluation metrics,
        effective classifier configuration, and feature metadata.

    Raises:
        ValueError: If the configured classifier alias is unknown or the data
            cannot be split or evaluated by scikit-learn.
    """
    configure_logging()
    logger = logging.getLogger(__name__)
    target_col = config.data.target_column
    test_size = config.data.test_size
    random_state = config.data.random_state

    try:
        # Features and target need to be separated for training and evaluation
        X, y = df.drop(columns=[target_col]), df[target_col]
        logger.info(
            "Starting training with %d rows, target '%s', and test_size %.3f.",
            len(df),
            target_col,
            test_size,
        )

        # A stratified split produces test data fit for model evaluation and promotion decisions
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        logger.info(
            "Train/test split complete: %d rows train, %d rows test.",
            len(X_train),
            len(X_test),
        )

        classifier, classifier_config = create_model(
            model_alias=config.model.classifier,
            model_params=config.model.classifier_params,
        )
        logger.info(
            "Using classifier '%s' with params %s.",
            config.model.classifier,
            classifier_config,
        )

        model_pipeline = build_classifier_pipeline(
            classifier=classifier,
            feature_params=config.feature_builder.feature_params,
            num_impute_strategy=config.preprocessing.numeric_impute_strategy,
            cat_impute_strategy=config.preprocessing.categorical_impute_strategy,
        )
        logger.info(
            "Built classifier pipeline with feature engineering, preprocessors and estimator."
        )

        # The model pipeline constructed above is fit on the training and evaluated on the test data.
        with Timer() as fit_timer:
            model_pipeline.fit(X_train, y_train)
        logger.info("Model pipeline fit completed in %.3f seconds.", fit_timer.duration)

        with Timer() as pred_timer:
            y_proba = model_pipeline.predict_proba(X_test)[:, 1]
        logger.info(
            "Prediction on test set completed in %.3f seconds.", pred_timer.duration
        )

        # Evaluation metrics are calcualted for logging and model promotion decisions
        metrics = evaluate_model(
            y_true=y_test,
            y_prob=y_proba,
            fit_time_sec=fit_timer.duration,
            pred_time_sec=pred_timer.duration,
            threshold=config.evaluation.threshold,
        )
        logger.info("Evaluation metrics: %s", vars(metrics))

        # Training artifacts and metadata are combined for reproducibility
        feature_names_out = (
            model_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        )
        metadata = {
            "training_config": config,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "feature_count": len(feature_names_out),
            "feature_names_in": list(X.columns),
            "feature_names_out": feature_names_out,
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        }
        return TrainingResult(
            trained_pipeline=model_pipeline,
            metrics=vars(metrics),
            classifier_config=classifier_config,
            metadata=metadata,
        )
    except Exception:
        logger.exception(
            "Training failed for target column '%s' and classifier '%s'.",
            target_col,
            config.model.classifier,
        )
        raise
