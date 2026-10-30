import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from churn_mlops.training.calibration import (
    fit_estimator_with_calibration,
    make_temporal_cv_splits,
)


@pytest.mark.unit
def test_temporal_cv_folds_only_use_historical_observations() -> None:
    data = pd.DataFrame(
        {
            "reference_date": pd.date_range("2026-01-01", periods=12)[::-1],
            "target": np.tile([0, 1], 6),
        }
    )

    splits = make_temporal_cv_splits(
        data,
        target_column="target",
        time_column="reference_date",
        n_splits=3,
    )

    assert len(splits) == 3
    for historical_indices, val_indices in splits:
        historical_dates = data.iloc[historical_indices]["reference_date"]
        val_dates = data.iloc[val_indices]["reference_date"]
        assert historical_dates.max() < val_dates.min()


@pytest.mark.unit
def test_temporal_cv_folds_keep_equal_timestamps_together() -> None:
    dates = pd.date_range("2026-01-01", periods=5).repeat(2)
    data = pd.DataFrame(
        {
            "reference_date": dates,
            "target": [0, 1] * 5,
        }
    )

    splits = make_temporal_cv_splits(
        data,
        target_column="target",
        time_column="reference_date",
        n_splits=2,
    )

    for historical_indices, val_indices in splits:
        historical_dates = data.iloc[historical_indices]["reference_date"]
        val_dates = data.iloc[val_indices]["reference_date"]
        assert historical_dates.max() < val_dates.min()
        assert set(historical_dates).isdisjoint(set(val_dates))
        assert data.iloc[historical_indices]["target"].nunique() == 2
        assert data.iloc[val_indices]["target"].nunique() == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("data", "target_column", "time_column", "n_splits", "error"),
    [
        (
            pd.DataFrame({"reference_date": pd.date_range("2026-01-01", periods=8)}),
            "target",
            "reference_date",
            2,
            "target column",
        ),
        (
            pd.DataFrame({"target": [0, 1] * 4}),
            "target",
            "reference_date",
            2,
            "time column",
        ),
        (
            pd.DataFrame(
                {
                    "target": [0, 1] * 4,
                    "reference_date": pd.date_range("2026-01-01", periods=8),
                }
            ),
            "target",
            "target",
            2,
            "must be different",
        ),
        (
            pd.DataFrame({"target": [0, 1] * 4, "reference_date": [None] * 8}),
            "target",
            "reference_date",
            2,
            "valid and non-missing",
        ),
        (
            pd.DataFrame(
                {
                    "target": [0, 1] * 4,
                    "reference_date": pd.date_range("2026-01-01", periods=8),
                }
            ),
            "target",
            "reference_date",
            1,
            "n_splits must be at least 2",
        ),
    ],
)
def test_temporal_cv_rejects_invalid_inputs(
    data: pd.DataFrame,
    target_column: str,
    time_column: str,
    n_splits: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        make_temporal_cv_splits(
            data,
            target_column=target_column,
            time_column=time_column,
            n_splits=n_splits,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "labels",
    [
        [0] * 6 + [1] * 4,
        [0, 1] * 3 + [0] * 4,
    ],
)
def test_temporal_cv_rejects_single_class_in_any_fold(labels: list[int]) -> None:
    data = pd.DataFrame(
        {
            "target": labels,
            "reference_date": pd.date_range("2026-01-01", periods=len(labels)),
        }
    )

    with pytest.raises(ValueError, match="both target classes"):
        make_temporal_cv_splits(
            data,
            target_column="target",
            time_column="reference_date",
            n_splits=2,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "classifier",
    [LogisticRegression(), LinearSVC(), SVC(kernel="rbf")],
    ids=["logistic-regression", "linear-svc", "svc"],
)
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_fit_estimator_with_calibration_supports_configured_methods(
    classifier: LogisticRegression | LinearSVC | SVC,
    method: str,
) -> None:
    timestamps = pd.date_range("2026-01-01", periods=40)
    labels = pd.Series(np.tile([0, 1], 20))
    train_df = pd.DataFrame(
        {
            "signal": np.arange(len(labels), dtype=float),
            "target": labels,
            "reference_date": timestamps,
        }
    )

    features = train_df[["signal"]]

    model = fit_estimator_with_calibration(
        estimator=classifier,
        train_df=train_df,
        target_column="target",
        time_column="reference_date",
        exclude_columns=["target", "reference_date"],
        method=method,
        n_splits=3,
    )

    assert isinstance(model, CalibratedClassifierCV)
    assert model.predict_proba(features).shape == (40, 2)


@pytest.mark.unit
@pytest.mark.parametrize("classifier", [LinearSVC(), SVC(kernel="rbf")])
def test_uncalibrated_svm_is_rejected(classifier: LinearSVC | SVC) -> None:
    train_df = pd.DataFrame(
        {
            "signal": np.arange(8, dtype=float),
            "target": [0, 1] * 4,
            "reference_date": pd.date_range("2026-01-01", periods=8),
        }
    )

    with pytest.raises(ValueError, match="configure calibration.method"):
        fit_estimator_with_calibration(
            estimator=classifier,
            train_df=train_df,
            target_column="target",
            time_column="reference_date",
            exclude_columns=["target", "reference_date"],
            method="none",
        )
