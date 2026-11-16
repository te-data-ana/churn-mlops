"""Optuna search for classifier hyperparameters using chronological CV."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.models import MODEL_CATALOG, build_classifier_pipeline, create_model
from churn_mlops.training.calibration import make_temporal_cv_splits
from churn_mlops.training.features import select_feature_columns
from churn_mlops.use_cases import UseCase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TuningResult:
    """Best hyperparameters and reproducibility metadata from an Optuna study."""

    best_params: dict[str, Any]
    best_score: float
    metric: str
    n_trials: int
    n_splits: int
    time_column: str
    random_state: int


def _suggest_parameters(
    trial: optuna.Trial,
    search_space: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Sample one catalog-defined parameter set for an Optuna trial."""
    suggestions: dict[str, Any] = {}
    for name, definition in search_space.items():
        kind = definition["type"]
        if kind == "int":
            suggestions[name] = trial.suggest_int(
                name,
                definition["low"],
                definition["high"],
                step=definition.get("step", 1),
                log=definition.get("log", False),
            )
        elif kind == "float":
            suggestions[name] = trial.suggest_float(
                name,
                definition["low"],
                definition["high"],
                step=definition.get("step"),
                log=definition.get("log", False),
            )
        elif kind == "categorical":
            suggestions[name] = trial.suggest_categorical(
                name,
                definition["choices"],
            )
        else:
            raise ValueError(
                f"Unsupported search-space type '{kind}' for parameter '{name}'."
            )
    return suggestions


def optimize_hyperparameters(
    config: TrainingConfig,
    train_df: pd.DataFrame,
    use_case: UseCase,
    n_jobs: int = 1,
) -> TuningResult:
    """Optimize the configured classifier on chronological CV folds.

    Args:
        config: Training configuration with tuning enabled.
        train_df: Training data split, including the configured time column.
            A separate out-of-time test split should be kept aside for final
            evaluation.
        use_case: Use-case contract for the training data.
        n_jobs: Total CPU job budget for parallel trials.
    Returns:
        Optimized catalog-defined parameters and study metadata.

    Raises:
        ValueError: If tuning settings, classifier search space, timestamps,
            labels, required model features, or folds are invalid.
    """
    if config.use_case != use_case.name:
        raise ValueError(
            f"Training config selects use case '{config.use_case}', but "
            f"'{use_case.name}' was supplied."
        )
    if type(n_jobs) is not int or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be a positive integer or -1.")
    tuning = config.tuning
    if not tuning.enabled:
        raise ValueError("Optuna tuning is disabled in the training configuration.")
    if tuning.n_trials < 1:
        raise ValueError("Tuning n_trials must be at least 1.")
    if tuning.metric not in {"roc_auc", "pr_auc"}:
        raise ValueError(
            f"Unsupported tuning metric '{tuning.metric}'. "
            "Choose 'roc_auc' or 'pr_auc'."
        )
    if config.model.classifier not in MODEL_CATALOG:
        raise ValueError(f"Unknown classifier '{config.model.classifier}'.")

    search_space = MODEL_CATALOG[config.model.classifier].get("search_space", {})
    if not search_space:
        raise ValueError(
            f"Classifier '{config.model.classifier}' does not define an Optuna "
            "search space."
        )

    target_column = use_case.target_column
    time_column = use_case.time_column
    train_features = select_feature_columns(
        df=train_df,
        feature_columns=use_case.feature_columns,
        data_name="Tuning",
    )
    folds = make_temporal_cv_splits(
        df=train_df,
        target_column=target_column,
        time_column=time_column,
        n_splits=tuning.n_splits,
    )
    metric = roc_auc_score if tuning.metric == "roc_auc" else average_precision_score

    def objective(trial: optuna.Trial) -> float:
        trial_params = _suggest_parameters(trial, search_space)
        fold_scores: list[float] = []
        for train_indices, validation_indices in folds:
            classifier, _ = create_model(
                model_alias=config.model.classifier,
                model_params=config.model.classifier_params | trial_params,
                n_jobs=1,
            )
            pipeline = build_classifier_pipeline(
                classifier=classifier,
                feature_builder=use_case.build_feature_builder(),
                num_impute_strategy=config.preprocessing.numeric_impute_strategy,
                cat_impute_strategy=config.preprocessing.categorical_impute_strategy,
            )
            fold_train_features = train_features.iloc[train_indices]
            fold_validation_features = train_features.iloc[validation_indices]
            fold_train_targets = train_df.iloc[train_indices][target_column]
            fold_validation_targets = train_df.iloc[validation_indices][target_column]
            pipeline.fit(
                X=fold_train_features,
                y=fold_train_targets,
            )
            if hasattr(pipeline, "predict_proba"):
                ranking_scores = pipeline.predict_proba(fold_validation_features)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                ranking_scores = pipeline.decision_function(fold_validation_features)
            else:
                raise ValueError(
                    f"Classifier '{config.model.classifier}' supports neither "
                    "predict_proba nor decision_function for ranking metrics."
                )
            fold_scores.append(float(metric(fold_validation_targets, ranking_scores)))
        return float(np.mean(fold_scores))

    logger.info(
        "Starting Optuna study for classifier '%s': %d trials, %d expanding "
        "window folds, metric '%s'.",
        config.model.classifier,
        tuning.n_trials,
        tuning.n_splits,
        tuning.metric,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=tuning.random_state),
    )
    with threadpool_limits(limits=1):
        study.optimize(func=objective, n_trials=tuning.n_trials, n_jobs=n_jobs)
    logger.info(
        "Optuna study completed: best %s=%.6f with params %s.",
        tuning.metric,
        study.best_value,
        study.best_params,
    )
    return TuningResult(
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        metric=tuning.metric,
        n_trials=tuning.n_trials,
        n_splits=tuning.n_splits,
        time_column=time_column,
        random_state=tuning.random_state,
    )
