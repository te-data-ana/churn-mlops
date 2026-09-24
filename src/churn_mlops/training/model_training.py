import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.evaluation import Timer, evaluate_model
from churn_mlops.models import build_classifier_pipeline, create_model


@dataclass
class TrainingResult:
    trained_pipeline: Pipeline
    metrics: dict[str, float]
    classifier_config: dict[str, Any]
    metadata: dict[str, Any]


def train_model(config: TrainingConfig, df: pd.DataFrame) -> TrainingResult:
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
