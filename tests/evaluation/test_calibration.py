import numpy as np
import pytest
from matplotlib.figure import Figure

from churn_mlops.evaluation.calibration import reliability_diagram


@pytest.mark.unit
def test_reliability_diagram_returns_a_figure() -> None:
    figure = reliability_diagram(
        y_true=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.1, 0.4, 0.6, 0.9]),
        n_bins=4,
    )

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
