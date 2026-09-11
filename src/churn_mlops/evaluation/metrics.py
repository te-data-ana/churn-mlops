from dataclasses import dataclass
from time import perf_counter

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

    def __enter__(self):
        self._start = perf_counter()
        return self

    def __exit__(self, *args):
        self.duration = perf_counter() - self._start


@dataclass
class ModelEvaluation:
    model: str
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier_score: float
    fit_time_sec: float
    pred_time_sec: float


def evaluate_model(
    model_name: str,
    y_true,
    y_prob,
    fit_time_sec: float | None = None,
    pred_time_sec: float | None = None,
    threshold: float = 0.5,
) -> ModelEvaluation:
    """
    Evaluate binary classification model.

    Parameters
    ----------
    model_name : str
        Name of the model.

    y_true : array-like
        Ground truth labels.

    y_prob : array-like
        Predicted positive class probabilities.

    threshold : float
        Probability threshold for positive class.

    Returns
    -------
    ModelEvaluation
        Evaluation metrics.
    """

    y_pred = (y_prob >= threshold).astype(int)

    return ModelEvaluation(
        model=model_name,
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
