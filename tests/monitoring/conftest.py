import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def valid_reference_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "age": [30, 45, 52],
            "tenure": [12, 24, 8],
            "usage_frequency": [10, 20, 15],
            "support_calls": [2, 1, 3],
            "payment_delay": [0, 5, 2],
            "last_interaction": [4, 10, 7],
            "total_spend": [200.0, 500.0, 350.0],
            "gender": ["Female", "Male", "Female"],
            "subscription_type": ["Basic", "Premium", "Standard"],
            "contract_length": ["Monthly", "Annual", "Quarterly"],
            "churn": [0, 1, 1],
            "reference_date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"]
            ),
            "predicted_probability": [0.2, 0.7, 0.8],
            "predicted_class": [0, 1, 1],
            "model_version": [1, 1, 1],
        }
    )
    return frame


@pytest.fixture
def valid_analysis_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "age": [31, 48],
            "tenure": [15, 19],
            "usage_frequency": [12, 22],
            "support_calls": [1, 2],
            "payment_delay": [1, 4],
            "last_interaction": [5, 9],
            "total_spend": [220.0, 470.0],
            "gender": ["Female", "Male"],
            "subscription_type": ["Basic", "Premium"],
            "contract_length": ["Monthly", "Annual"],
            "reference_date": pd.to_datetime(["2026-02-01", "2026-02-02"]),
            "predicted_probability": [0.35, 0.75],
            "predicted_class": [0, 1],
            "model_version": [1, 1],
        }
    )
    return frame


@pytest.fixture
def valid_prediction_log_path(tmp_path: Path) -> Path:
    path = tmp_path / "predictions.jsonl"
    records = [
        {
            "event": "prediction",
            "request_id": "req-1",
            "reference_date": "2026-02-01T00:00:00Z",
            "latency_ms": 12.5,
            "model_name": "churn-model",
            "model_alias": "champion",
            "model_version": 1,
            "predicted_class": 0,
            "predicted_probability": 0.35,
            "threshold": 0.5,
            "features": {
                "age": 31,
                "tenure": 15,
                "usage_frequency": 12,
                "support_calls": 1,
                "payment_delay": 1,
                "last_interaction": 5,
                "total_spend": 220.0,
                "gender": "Female",
                "subscription_type": "Basic",
                "contract_length": "Monthly",
            },
        },
        {
            "event": "prediction",
            "request_id": "req-2",
            "reference_date": "2026-02-02T00:00:00Z",
            "latency_ms": 18.0,
            "model_name": "churn-model",
            "model_alias": "champion",
            "model_version": 1,
            "predicted_class": 1,
            "predicted_probability": 0.75,
            "threshold": 0.5,
            "features": {
                "age": 48,
                "tenure": 19,
                "usage_frequency": 22,
                "support_calls": 2,
                "payment_delay": 4,
                "last_interaction": 9,
                "total_spend": 470.0,
                "gender": "Male",
                "subscription_type": "Premium",
                "contract_length": "Annual",
            },
        },
    ]
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return path


@pytest.fixture
def valid_error_log_path(tmp_path: Path) -> Path:
    path = tmp_path / "prediction_errors.jsonl"
    records = [
        {
            "event": "prediction_error",
            "request_id": "err-1",
            "reference_date": "2026-02-03T00:00:00Z",
            "latency_ms": 55.0,
            "model_name": "churn-model",
            "model_alias": "champion",
            "model_version": 1,
            "error_type": "ValueError",
            "error_message": "bad input",
            "traceback": "Traceback ...",
            "features": {
                "age": 40,
                "tenure": 10,
                "usage_frequency": 5,
                "support_calls": 2,
                "payment_delay": 0,
                "last_interaction": 3,
                "total_spend": 300.0,
                "gender": "Female",
                "subscription_type": "Standard",
                "contract_length": "Quarterly",
            },
        }
    ]
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return path
