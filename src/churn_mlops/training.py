import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_mlops.config import configure_logging, load_config
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
    training_file: str = "customer_churn_dataset-training.csv",
    index_col: str = "customerid",
    experiment_name: str = "test",
) -> TrainingResult:
    """Run the configured training, tracking, registration, and promotion flow.

    Args:
        config_file: Training configuration YAML filename.
        training_file: Raw training data CSV filename.
        index_col: Column to use as the training DataFrame index.
        experiment_name: Local MLflow experiment name.

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

    config, config_file_path = load_config(config_file)
    classifier_alias = config.model.classifier
    eval_threshold = config.evaluation.threshold

    run_name = f"{classifier_alias}_{eval_threshold!s}"
    experiment_id = setup_local_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)

    # Load and validate the training data
    df = load_raw_data(training_file, index_col)
    df = validate_training_data(df)

    logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(str(experiment))

    # Start an MLflow run for the training job and log the results
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        result = train(config, df)
        model_info = log_experiment_result(result, config, config_file_path)

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
        else:
            logger.info("No promotion of candidate model:")
            logger.info(promotion_decision.reason)

    return result


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

    # Features and target need to be separated for training and evaluation
    X, y = df.drop(columns=[config.data.target_column]), df[config.data.target_column]

    # A stratified split produces test data fit for model evaluation and promotion decisions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )

    classifier, classifier_config = create_model(
        model_alias=config.model.classifier, model_params=config.model.classifier_params
    )

    model_pipeline = build_classifier_pipeline(
        classifier=classifier,
        feature_params=config.feature_builder.feature_params,
        num_impute_strategy=config.preprocessing.numeric_impute_strategy,
        cat_impute_strategy=config.preprocessing.categorical_impute_strategy,
    )

    # The model pipeline constructed above is fit on the training and evaluated on the test data.
    with Timer() as fit_timer:
        model_pipeline.fit(X_train, y_train)
    with Timer() as pred_timer:
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    # Evaluation metrics are calcualted for logging and model promotion decisions
    metrics = evaluate_model(
        y_true=y_test,
        y_prob=y_proba,
        fit_time_sec=fit_timer.duration,
        pred_time_sec=pred_timer.duration,
        threshold=config.evaluation.threshold,
    )

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
