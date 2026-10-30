"""Classification threshold selection helpers."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import f1_score


def select_f1_threshold(
    y_true: ArrayLike,
    y_prob: ArrayLike,
) -> float:
    """Select the threshold maximizing F1, preferring higher thresholds on ties.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        The selected probability threshold.

    Raises:
        ValueError: If labels do not contain both binary classes or
            probabilities are invalid.
    """
    label_array = np.asarray(y_true)
    prob_array = np.asarray(y_prob, dtype=float)
    if np.unique(label_array).size != 2 or not set(np.unique(label_array)).issubset(
        {0, 1}
    ):
        raise ValueError("Threshold selection data must contain both binary classes.")
    if (
        prob_array.ndim != 1
        or prob_array.size != label_array.size
        or not np.isfinite(prob_array).all()
        or ((prob_array < 0) | (prob_array > 1)).any()
    ):
        raise ValueError("Threshold probabilities must be finite values in [0, 1].")

    thresholds = np.unique(np.concatenate(([0.0], prob_array, [1.0])))
    scores = np.asarray(
        [
            f1_score(
                label_array,
                prob_array >= threshold,
                zero_division=0,
            )
            for threshold in thresholds
        ]
    )
    best_score = float(scores.max())
    best_threshold = float(thresholds[scores == best_score][-1])
    return best_threshold
