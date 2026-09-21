from dataclasses import dataclass
from time import perf_counter
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
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
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    fit_time_sec: float | None
    pred_time_sec: float | None


def evaluate_model(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    fit_time_sec: float | None = None,
    pred_time_sec: float | None = None,
    threshold: float = 0.5,
) -> ModelEvaluation:
    """Evaluate a binary classifier from labels and positive probabilities.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.
        fit_time_sec: Optional time spent fitting the model, in seconds.
        pred_time_sec: Optional time spent generating predictions, in seconds.
        threshold: Probability threshold used to derive predicted classes.

    Returns:
        ModelEvaluation containing ROC AUC, PR AUC, classification metrics,
        Brier score, and the supplied timing values.

    Raises:
        ValueError: If the supplied labels or probabilities are invalid for a
            scikit-learn metric.
    """

    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return ModelEvaluation(
        roc_auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        brier_score=brier_score_loss(y_true, y_prob),
        fit_time_sec=fit_time_sec,
        pred_time_sec=pred_time_sec,
    )
