from pathlib import Path

import pandas as pd
import pytest

from churn_mlops.monitoring import core


@pytest.fixture
def reference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [30, 45],
            "tenure": [12, 24],
            "usage_frequency": [10, 20],
            "support_calls": [2, 1],
            "payment_delay": [0, 5],
            "last_interaction": [4, 10],
            "total_spend": [200.0, 500.0],
            "gender": ["Female", "Male"],
            "subscription_type": ["Basic", "Premium"],
            "contract_length": ["Monthly", "Annual"],
            "churn": [0, 1],
            "reference_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "predicted_probability": [0.2, 0.8],
            "predicted_class": [0, 1],
            "model_version": [1, 1],
        },
        index=pd.Index([101, 102]),
    )


@pytest.fixture
def api_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "reference_date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
            "model_version": [1, 1],
            "predicted_probability": [0.3, 0.7],
            "predicted_class": [0, 1],
            "latency_ms": [10.0, 30.0],
        }
    )
    for column in core.FEATURE_COLUMNS:
        frame[column] = pd.NA
    return frame


class _FakeResult:
    def __init__(self, columns: list[tuple[str, str]]) -> None:
        self.frame = pd.DataFrame(
            [
                [
                    pd.Timestamp("2026-02-01"),
                    *[False if name == "alert" else 0.75 for _, name in columns[1:]],
                ]
            ],
            columns=pd.MultiIndex.from_tuples(columns),
        )

    def to_df(self) -> pd.DataFrame:
        return self.frame


class _FakeDriftCalculator:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, reference: pd.DataFrame) -> "_FakeDriftCalculator":
        self.reference = reference
        return self

    def calculate(self, analysis: pd.DataFrame) -> _FakeResult:
        return _FakeResult([("chunk", "start_date"), ("drift", "alert")])

    def estimate(self, analysis: pd.DataFrame) -> _FakeResult:
        return _FakeResult([("chunk", "start_date"), ("roc_auc", "value")])


def test_missing_error_log_is_treated_as_no_errors(tmp_path: Path) -> None:
    frame = core._event_log_to_frame(
        tmp_path / "not-created-yet.jsonl", event="prediction_error"
    )

    assert frame.empty


def test_run_monitoring_requires_exactly_one_analysis_source(tmp_path: Path) -> None:
    try:
        core.run_monitoring(tmp_path / "reference.csv")
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("Expected missing analysis source to be rejected.")
