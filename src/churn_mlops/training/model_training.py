import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
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


def train_model(
    config: TrainingConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> TrainingResult:
    """Fit and evaluate a configured classifier pipeline.

    Args:
        config: Training, feature, preprocessing, model, evaluation and
            MLflow registration settings.
        train_df: Validated training DataFrame containing features and target.
        test_df: Validated test DataFrame containing features and target.

    Returns:
        TrainingResult containing the fitted pipeline, evaluation metrics,
        effective classifier configuration, and feature metadata.

    Raises:
        ValueError: If the configured classifier alias is unknown.
    """
    logger = logging.getLogger(__name__)
    target_col = config.data.target_column

    try:
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        logger.info(
            "Starting training using prepared train/test split with %d"
            "train rows and %d test rows for target '%s'.",
            len(train_df),
            len(test_df),
            target_col,
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
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "feature_count": len(feature_names_out),
            "feature_names_in": list(X_train.columns),
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
