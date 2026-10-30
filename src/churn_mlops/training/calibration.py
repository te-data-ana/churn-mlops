"""Time-aware calibration fitting and fold construction."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit


def make_temporal_cv_splits(
    df: pd.DataFrame,
    *,
    target_column: str,
    time_column: str,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding historical folds while keeping timestamp groups intact.

    Args:
        df: Training dataframe containing the target and timestamp columns.
        target_column: Column containing the target labels.
        time_column: Column defining chronological order.
        n_splits: Number of expanding-window folds.

    Returns:
        Scikit-learn-compatible train/calibration index pairs.

    Raises:
        ValueError: If columns, timestamps, labels, or split count are invalid.
    """
    if target_column not in df.columns:
        raise ValueError(f"Temporal CV target column '{target_column}' is missing.")
    if time_column not in df.columns:
        raise ValueError(f"Temporal CV time column '{time_column}' is missing.")
    if target_column == time_column:
        raise ValueError("Temporal CV target and time columns must be different.")
    timestamp_series = pd.to_datetime(
        df[time_column],
        errors="coerce",
        utc=True,
    )
    if timestamp_series.isna().any():
        raise ValueError("Temporal CV timestamps must be valid and non-missing.")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("Temporal CV n_splits must be at least 2.")
    unique_timestamps = pd.Index(timestamp_series.drop_duplicates()).sort_values()
    if len(unique_timestamps) <= n_splits:
        raise ValueError(
            "Temporal CV requires more distinct timestamps than "
            f"n_splits ({len(unique_timestamps)} timestamps, "
            f"{n_splits} splits)."
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    timestamp_array = timestamp_series.to_numpy()
    labels = df[target_column].to_numpy()
    for train_times, val_times in splitter.split(unique_timestamps):
        train_unique = unique_timestamps[train_times]
        val_unique = unique_timestamps[val_times]
        train_indices = np.flatnonzero(np.isin(timestamp_array, train_unique))
        val_indices = np.flatnonzero(np.isin(timestamp_array, val_unique))
        if (
            pd.Series(labels[train_indices]).nunique() < 2
            or pd.Series(labels[val_indices]).nunique() < 2
        ):
            raise ValueError("Every temporal CV fold must contain both target classes.")
        splits.append((train_indices, val_indices))
    return splits


def fit_estimator_with_calibration(
    estimator: BaseEstimator,
    train_df: pd.DataFrame,
    *,
    target_column: str,
    time_column: str,
    exclude_columns: Sequence[str] = (),
    method: str = "none",
    n_splits: int = 4,
) -> BaseEstimator:
    """Fit an estimator, optionally using historical-only calibration folds.

    Args:
        estimator: Unfitted scikit-learn estimator or pipeline.
        train_df: Full training data containing target and time columns.
        target_column: Column containing the target labels.
        time_column: Column defining chronological order.
        method: ``none``, ``sigmoid``, or ``isotonic``.
        n_splits: Number of expanding-window calibration folds.
        exclude_columns: Optional non-feature columns to exclude from fitting.

    Returns:
        A fitted estimator, wrapped in ``CalibratedClassifierCV`` when enabled.

    Raises:
        ValueError: If the method is unsupported or temporal folds are invalid.
    """
    if target_column not in train_df.columns:
        raise ValueError(f"Temporal CV target column '{target_column}' is missing.")

    X_train = train_df.drop(columns=exclude_columns, errors="ignore")
    y_train = train_df[target_column]

    if method == "none":
        if not hasattr(estimator, "predict_proba"):
            raise ValueError(
                f"Classifier '{estimator.__class__.__name__}' does not expose "
                "predict_proba; configure calibration.method as 'sigmoid' or "
                "'isotonic'."
            )
        return estimator.fit(X_train, y_train)
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError(
            "Calibration method must be one of 'none', 'sigmoid', or 'isotonic'."
        )

    splits = make_temporal_cv_splits(
        df=train_df,
        target_column=target_column,
        time_column=time_column,
        n_splits=n_splits,
    )
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=splits,
        ensemble=True,
    )
    return calibrated.fit(X_train, y_train)
