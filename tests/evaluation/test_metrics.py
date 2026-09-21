import numpy as np
import pytest

from churn_mlops.evaluation import metrics


@pytest.mark.unit
def test_timer_records_elapsed_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    # Freeze the clock so the context manager's measured duration is deterministic.
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(metrics, "perf_counter", lambda: next(clock))

    with metrics.Timer() as timer:
        pass

    assert timer.duration == pytest.approx(0.25)


@pytest.mark.unit
def test_evaluate_model_with_perfect_predictions_returns_perfect_metrics() -> None:
    # Use perfectly separated probabilities so every classification metric is 1.0.
    evaluation = metrics.evaluate_model(
        y_true=np.array([0, 0, 1, 1]),
        y_prob=np.array([0.1, 0.4, 0.6, 0.9]),
        fit_time_sec=1.2,
        pred_time_sec=0.3,
    )

    # Verify the ranking, classification, calibration, and timing metrics.
    assert evaluation.roc_auc == pytest.approx(1.0)
    assert evaluation.pr_auc == pytest.approx(1.0)
    assert evaluation.accuracy == pytest.approx(1.0)
    assert evaluation.precision == pytest.approx(1.0)
    assert evaluation.recall == pytest.approx(1.0)
    assert evaluation.f1 == pytest.approx(1.0)
    assert evaluation.brier_score == pytest.approx(0.085)
    assert evaluation.fit_time_sec == 1.2
    assert evaluation.pred_time_sec == 0.3


@pytest.mark.unit
def test_evaluate_model_uses_threshold_for_classification_metrics() -> None:
    # The custom threshold changes probabilities into two positive predictions.
    evaluation = metrics.evaluate_model(
        y_true=np.array([0, 1, 1, 0]),
        y_prob=np.array([0.2, 0.55, 0.8, 0.9]),
        threshold=0.6,
    )

    # Check metrics that depend on the thresholded class predictions.
    assert evaluation.accuracy == pytest.approx(0.5)
    assert evaluation.precision == pytest.approx(0.5)
    assert evaluation.recall == pytest.approx(0.5)
    assert evaluation.f1 == pytest.approx(0.5)
