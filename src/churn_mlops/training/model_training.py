import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.evaluation import (
    Timer,
    evaluate_model,
    reliability_diagram,
    select_f1_threshold,
)
from churn_mlops.models import build_classifier_pipeline, create_model
from churn_mlops.training.calibration import fit_estimator_with_calibration


@dataclass
class TrainingResult:
    """Fitted estimator and reproducible evaluation outputs from training."""

    trained_pipeline: BaseEstimator
    metrics: dict[str, float]
    classifier_config: dict[str, Any]
    feature_names_in: list[str]
    feature_names_out: list[str]
    threshold: float
    threshold_data: str
    calibration_figure: Figure


def train_model(
    config: TrainingConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_params_override: Mapping[str, Any] | None = None,
    exclude_columns: Sequence[str] = (),
) -> TrainingResult:
    """Fit and evaluate a configured classifier pipeline.

    Args:
        config: Training, feature, preprocessing, model, calibration, threshold,
            promotion, and MLflow registration settings.
        train_df: Validated training DataFrame containing features and target.
        test_df: Validated test DataFrame containing features and target.
        model_params_override: Optional model parameters overriding configured
            values for this fit.
        exclude_columns: Optional non-feature columns to exclude from fitting
            and evaluation. The configured data time column is always excluded
            from model features.

    Returns:
        TrainingResult containing the fitted pipeline, evaluation metrics,
        effective classifier configuration, feature names, and threshold
        selection details.

    Raises:
        ValueError: If the configured classifier alias is unknown.
    """
    logger = logging.getLogger(__name__)
    target_column = config.data.target_column
    time_column = config.data.time_column
    exclude_columns = [target_column, time_column, *exclude_columns]

    try:
        X_train = train_df.drop(columns=exclude_columns, errors="ignore")
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=exclude_columns, errors="ignore")
        y_test = test_df[target_column]
        if y_train.nunique() != 2:
            raise ValueError("Training data must contain both target classes.")
        logger.info(
            "Starting training using prepared train/test split with %d "
            "train rows and %d test rows for target '%s'.",
            len(train_df),
            len(test_df),
            target_column,
        )

        classifier, classifier_config = create_model(
            model_alias=config.model.classifier,
            model_params=(
                config.model.classifier_params | (model_params_override or {})
            ),
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

        with Timer() as fit_timer:
            trained_model = fit_estimator_with_calibration(
                estimator=model_pipeline,
                train_df=train_df,
                target_column=target_column,
                time_column=time_column,
                exclude_columns=exclude_columns,
                method=config.calibration.method,
                n_splits=config.calibration.n_splits,
            )
        logger.info("Model pipeline fit completed in %.3f seconds.", fit_timer.duration)

        with Timer() as pred_timer:
            y_proba = trained_model.predict_proba(X_test)[:, 1]
        logger.info(
            "Prediction on test set completed in %.3f seconds.", pred_timer.duration
        )

        # Threshold and evaluation metrics are calcualted on out-of-time test data
        threshold = select_f1_threshold(y_test, y_proba)
        logger.info(
            "Selected threshold %.6f using %s on out-of-time test data.",
            threshold,
            config.threshold.strategy,
        )

        metrics = evaluate_model(
            y_true=y_test,
            y_prob=y_proba,
            threshold=threshold,
            calibration_bins=config.calibration.n_bins,
            fit_time_sec=fit_timer.duration,
            pred_time_sec=pred_timer.duration,
        )
        logger.info("Evaluation metrics: %s", vars(metrics))

        fitted_base_pipeline = (
            trained_model.calibrated_classifiers_[0].estimator
            if isinstance(trained_model, CalibratedClassifierCV)
            else trained_model
        )
        feature_names_out = (
            fitted_base_pipeline.named_steps["preprocessor"]
            .get_feature_names_out()
            .tolist()
        )
        calibration_figure = reliability_diagram(
            y_test,
            y_proba,
            n_bins=config.calibration.n_bins,
        )
        return TrainingResult(
            trained_pipeline=trained_model,
            metrics=vars(metrics),
            classifier_config=classifier_config,
            feature_names_in=list(X_train.columns),
            feature_names_out=feature_names_out,
            threshold=threshold,
            threshold_data="out_of_time_test",
            calibration_figure=calibration_figure,
        )
    except Exception:
        logger.exception(
            "Training failed for target column '%s' and classifier '%s'.",
            target_column,
            config.model.classifier,
        )
        raise
