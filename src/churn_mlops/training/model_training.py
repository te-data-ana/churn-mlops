import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from joblib import effective_n_jobs
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from threadpoolctl import threadpool_limits

from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.evaluation import (
    Timer,
    evaluate_model,
    reliability_diagram,
    select_f1_threshold,
)
from churn_mlops.models import (
    MODEL_CATALOG,
    build_classifier_pipeline,
    create_model,
)
from churn_mlops.training.calibration import fit_estimator_with_calibration
from churn_mlops.training.features import select_feature_columns
from churn_mlops.use_cases import UseCase


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
    use_case: UseCase,
    config: TrainingConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_params_override: Mapping[str, Any] | None = None,
    calibration_method: str = "none",
    n_jobs: int = 1,
) -> TrainingResult:
    """Fit and evaluate a configured classifier pipeline.

    Args:
        use_case: Explicit data contract to be used for model training.
        config: Training, feature, preprocessing, model, calibration-fold,
            threshold, promotion, and registry settings.
        train_df: Validated training DataFrame containing features and target.
        test_df: Validated test DataFrame containing features and target.
        model_params_override: Optional model parameters overriding configured
            values for this fit.
        calibration_method: Probability calibration method for this fit.
        n_jobs: Total CPU job budget for tuning, fitting, or calibration folds.
    Returns:
        TrainingResult containing the fitted pipeline, evaluation metrics,
        effective classifier configuration, feature names, and threshold
        selection details.

    Raises:
        ValueError: If required model features are missing or the
            classifier/labels are invalid.
    """
    if not isinstance(calibration_method, str) or calibration_method not in {
        "none",
        "sigmoid",
        "isotonic",
    }:
        raise ValueError(
            "Calibration method must be one of 'none', 'sigmoid', or 'isotonic'."
        )
    if type(n_jobs) is not int or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be a positive integer or -1.")
    logger = logging.getLogger(__name__)
    target_column = use_case.target_column
    time_column = use_case.time_column

    try:
        X_train = select_feature_columns(
            df=train_df,
            feature_columns=use_case.feature_columns,
            data_name="Training",
        )
        y_train = train_df[target_column]
        X_test = select_feature_columns(
            df=test_df,
            feature_columns=use_case.feature_columns,
            data_name="Test",
        )
        y_test = test_df[target_column]
        if y_train.nunique() != 2:
            raise ValueError("Training data must contain both target classes.")
        logger.info(
            "Prepared training data: %d rows and %d features; out-of-time test "
            "data: %d rows and %d features (target '%s').",
            len(X_train),
            X_train.shape[1],
            len(X_test),
            X_test.shape[1],
            target_column,
        )

        classifier, classifier_config = create_model(
            model_alias=config.model.classifier,
            model_params=(
                config.model.classifier_params | (model_params_override or {})
            ),
            n_jobs=(n_jobs if calibration_method == "none" else 1),
        )
        logger.info(
            "Using classifier '%s' with params %s.",
            config.model.classifier,
            classifier_config,
        )

        model_pipeline = build_classifier_pipeline(
            classifier=classifier,
            feature_builder=use_case.build_feature_builder(),
            num_impute_strategy=config.preprocessing.numeric_impute_strategy,
            cat_impute_strategy=config.preprocessing.categorical_impute_strategy,
        )
        logger.info(
            "Starting model fit for classifier '%s' with calibration method '%s'.",
            config.model.classifier,
            calibration_method,
        )

        thread_limit = (
            1
            if calibration_method != "none"
            or MODEL_CATALOG[config.model.classifier].get("supports_n_jobs", False)
            else effective_n_jobs(n_jobs)
        )
        with threadpool_limits(limits=thread_limit), Timer() as fit_timer:
            trained_model = fit_estimator_with_calibration(
                estimator=model_pipeline,
                train_df=train_df,
                target_column=target_column,
                time_column=time_column,
                feature_columns=use_case.feature_columns,
                method=calibration_method,
                n_splits=config.calibration.n_splits,
                n_jobs=n_jobs,
            )
        logger.info("Model pipeline fit completed in %.3f seconds.", fit_timer.duration)

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
        logger.info(
            "With %d input features resulting in a total of %d transformed features.",
            len(X_train.columns),
            len(feature_names_out),
        )

        with Timer() as pred_timer:
            y_proba = trained_model.predict_proba(X_test)[:, 1]
        logger.info(
            "Prediction on test set completed in %.3f seconds.", pred_timer.duration
        )

        logger.info(
            "Selecting decision threshold with strategy '%s' on out-of-time test data.",
            config.threshold.strategy,
        )
        threshold = select_f1_threshold(y_test, y_proba)
        logger.info(
            "Selected decision threshold %.6f.",
            threshold,
        )

        logger.info("Evaluating model on out-of-time test data.")
        metrics = evaluate_model(
            y_true=y_test,
            y_prob=y_proba,
            threshold=threshold,
            calibration_bins=config.calibration.n_bins,
            fit_time_sec=fit_timer.duration,
            pred_time_sec=pred_timer.duration,
        )
        logger.info("Evaluation metrics: %s", vars(metrics))
        calibration_figure = reliability_diagram(
            y_test,
            y_proba,
            n_bins=config.calibration.n_bins,
        )
        logger.info(
            "Prepared calibration reliability diagram with %d bins.",
            config.calibration.n_bins,
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
