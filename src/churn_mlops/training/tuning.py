"""Optuna search for classifier hyperparameters using chronological CV."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.models import MODEL_CATALOG, build_classifier_pipeline, create_model

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


def _build_expanding_window_folds(
    train_df: pd.DataFrame,
    target_column: str,
    time_column: str,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window folds that keep equal timestamps together."""
    if time_column not in train_df.columns:
        raise ValueError(
            f"Temporal tuning requires time column '{time_column}' in the "
            "training split."
        )
    if target_column not in train_df.columns:
        raise ValueError(f"Training split is missing target column '{target_column}'.")
    if time_column == target_column:
        raise ValueError("The temporal tuning column cannot be the target column.")
    if n_splits < 2:
        raise ValueError("Tuning n_splits must be at least 2.")

    timestamps = pd.to_datetime(train_df[time_column], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError(
            f"Temporal tuning column '{time_column}' contains missing or invalid "
            "timestamps."
        )

    unique_timestamps = pd.Index(timestamps.unique()).sort_values()
    if len(unique_timestamps) <= n_splits:
        raise ValueError(
            f"Temporal tuning requires more than {n_splits} distinct timestamps "
            f"in '{time_column}', found {len(unique_timestamps)}."
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    labels = train_df[target_column].reset_index(drop=True)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_times, validation_times in splitter.split(unique_timestamps):
        train_mask = timestamps.isin(unique_timestamps[train_times]).to_numpy()
        validation_mask = timestamps.isin(
            unique_timestamps[validation_times]
        ).to_numpy()
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        if labels.iloc[train_indices].nunique() < 2:
            raise ValueError(
                "An expanding-window training fold contains fewer than two "
                "target classes."
            )
        if labels.iloc[validation_indices].nunique() < 2:
            raise ValueError(
                "An expanding-window validation fold contains fewer than two "
                "target classes; ROC AUC and PR AUC require both classes."
            )
        folds.append((train_indices, validation_indices))
    return folds


def optimize_hyperparameters(
    config: TrainingConfig,
    train_df: pd.DataFrame,
) -> TuningResult:
    """Optimize the configured classifier on chronological CV folds.

    Args:
        config: Training configuration with tuning enabled.
        train_df: Validated training split, including the configured time
            column. The held-out test split is deliberately not accepted.

    Returns:
        Optimized catalog-defined parameters and study metadata.

    Raises:
        ValueError: If tuning settings, classifier search space, timestamps,
            labels, or folds are invalid.
    """
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

    target_column = config.data.target_column
    folds = _build_expanding_window_folds(
        train_df=train_df,
        target_column=target_column,
        time_column=tuning.time_column,
        n_splits=tuning.n_splits,
    )
    feature_columns = [
        column
        for column in train_df.columns
        if column not in {target_column, tuning.time_column}
    ]
    metric = roc_auc_score if tuning.metric == "roc_auc" else average_precision_score

    def objective(trial: optuna.Trial) -> float:
        trial_params = _suggest_parameters(trial, search_space)
        fold_scores: list[float] = []
        for train_indices, validation_indices in folds:
            classifier, _ = create_model(
                model_alias=config.model.classifier,
                model_params=config.model.classifier_params | trial_params,
            )
            pipeline = build_classifier_pipeline(
                classifier=classifier,
                feature_params=config.feature_builder.feature_params,
                num_impute_strategy=config.preprocessing.numeric_impute_strategy,
                cat_impute_strategy=config.preprocessing.categorical_impute_strategy,
            )
            fold_train = train_df.iloc[train_indices]
            fold_validation = train_df.iloc[validation_indices]
            pipeline.fit(
                X=fold_train.loc[:, feature_columns],
                y=fold_train[target_column],
            )
            probabilities = pipeline.predict_proba(
                X=fold_validation.loc[:, feature_columns]
            )[:, 1]
            fold_scores.append(
                float(metric(fold_validation[target_column], probabilities))
            )
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=tuning.random_state),
    )
    logger.info(
        "Starting Optuna study for classifier '%s': %d trials, %d expanding "
        "window folds, metric '%s'.",
        config.model.classifier,
        tuning.n_trials,
        tuning.n_splits,
        tuning.metric,
    )
    study.optimize(func=objective, n_trials=tuning.n_trials)
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
        time_column=tuning.time_column,
        random_state=tuning.random_state,
    )
