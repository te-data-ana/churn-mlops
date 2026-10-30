from dataclasses import dataclass
from time import perf_counter
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Timer:
    duration: float | None = None

    def __enter__(self) -> Self:
        """Start timing and return this context manager."""
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Store elapsed monotonic time when leaving the context."""
        self.duration = perf_counter() - self._start


@dataclass
class ModelEvaluation:
    """Ranking, classification, calibration, and timing metrics."""

    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    calibration_error: float
    fit_time_sec: float | None
    pred_time_sec: float | None


def evaluate_model(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    threshold: float,
    calibration_bins: int = 10,
    fit_time_sec: float | None = None,
    pred_time_sec: float | None = None,
) -> ModelEvaluation:
    """Evaluate a binary classifier from labels and positive probabilities.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        threshold: Probability threshold used to derive predicted classes.
        calibration_bins: Number of equally sized confidence bins used for
            expected calibration error.
        fit_time_sec: Optional time spent fitting the model, in seconds.
        pred_time_sec: Optional time spent generating predictions, in seconds.

    Returns:
        ModelEvaluation containing ROC AUC, PR AUC, classification metrics,
        Brier score, expected calibration error, and the supplied timing values.

    Raises:
        ValueError: If the supplied labels or probabilities are invalid for a
            scikit-learn metric.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be a positive integer.")

    y_prob = np.asarray(y_prob, dtype=float)
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Predicted probabilities must be finite values in [0, 1].")
    y_pred = (y_prob >= threshold).astype(int)
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        y_prob,
        n_bins=calibration_bins,
        strategy="uniform",
    )
    bin_counts = np.histogram(
        y_prob,
        bins=np.linspace(0.0, 1.0, calibration_bins + 1),
    )[0]
    nonempty_counts = bin_counts[bin_counts > 0]
    calibration_error = float(
        np.average(
            np.abs(fraction_positive - mean_predicted),
            weights=nonempty_counts,
        )
    )

    return ModelEvaluation(
        roc_auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        brier_score=brier_score_loss(y_true, y_prob),
        calibration_error=calibration_error,
        fit_time_sec=fit_time_sec,
        pred_time_sec=pred_time_sec,
    )
