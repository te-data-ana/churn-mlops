import numpy as np
import pytest

from churn_mlops.evaluation.threshold import select_f1_threshold


@pytest.mark.unit
def test_select_f1_threshold_returns_best_selection_threshold() -> None:
    threshold = select_f1_threshold(
        y_true=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.1, 0.4, 0.6, 0.7]),
    )

    assert threshold == pytest.approx(0.6)
