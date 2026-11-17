import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from churn_mlops.serving.prediction_logger import PredictionLogger
from churn_mlops.serving.predictor import PredictionResult
from churn_mlops.use_cases import get_use_case
from churn_mlops.use_cases.churn.serving import ChurnInputFeatures

CHURN_USE_CASE = get_use_case("churn")
ALTERNATE_USE_CASE = replace(CHURN_USE_CASE, time_column="event_date")


def test_log_prediction_writes_prediction_event(
    tmp_path: Path,
    sample_input: ChurnInputFeatures,
    sample_prediction_result: PredictionResult,
) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    error_path = tmp_path / "prediction_errors.jsonl"

    logger = PredictionLogger(
        prediction_log_path=pred_path,
        error_log_path=error_path,
        log_features=True,
        use_case=ALTERNATE_USE_CASE,
    )

    prediction = sample_prediction_result

    logger.log_prediction(
        request_id="req-123",
        latency_ms=15.2,
        prediction=prediction,
        features=sample_input,
    )

    record = json.loads(pred_path.read_text().splitlines()[0])

    assert record["request_id"] == "req-123"
    assert record["predicted_class"] == sample_prediction_result.predicted_class
    assert (
        record["predicted_probability"]
        == sample_prediction_result.predicted_probability
    )
    assert record["event_date"]
    assert "reference_date" not in record
    assert record["features"] == sample_input.model_dump()


def test_log_error_writes_error_event_with_features(
    tmp_path: Path,
    sample_input: ChurnInputFeatures,
) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    error_path = tmp_path / "prediction_errors.jsonl"

    logger = PredictionLogger(
        prediction_log_path=pred_path,
        error_log_path=error_path,
        log_features=True,
        use_case=ALTERNATE_USE_CASE,
    )

    class FakePredictor:
        use_case = ALTERNATE_USE_CASE

        def __init__(self):
            self.metadata = SimpleNamespace(
                model_name="test_model",
                model_alias="champion",
                model_version=1,
            )

        @staticmethod
        def _timestamp():
            return "2026-01-01T00:00:00"

    exc = ValueError("invalid input")

    logger.log_error(
        request_id="req-1",
        latency_ms=12.3,
        predictor=FakePredictor(),
        exception=exc,
        traceback_text="traceback content",
        features=sample_input,
    )

    lines = error_path.read_text().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["event"] == "prediction_error"
    assert record["request_id"] == "req-1"
    assert record["event_date"] == "2026-01-01T00:00:00"
    assert "reference_date" not in record
    assert record["error_type"] == "ValueError"
    assert record["error_message"] == "invalid input"
    assert record["traceback"] == "traceback content"
    assert record["features"] == sample_input.model_dump()


def test_prediction_logger_init_creates_parent_directories(tmp_path: Path) -> None:
    pred_path = tmp_path / "nested" / "predictions.jsonl"
    error_path = tmp_path / "nested" / "prediction_errors.jsonl"

    PredictionLogger(
        prediction_log_path=pred_path,
        error_log_path=error_path,
        log_features=False,
        use_case=CHURN_USE_CASE,
    )

    assert pred_path.parent.exists()
    assert error_path.parent.exists()


def test_append_jsonl_appends_multiple_records(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"

    PredictionLogger._append_jsonl(path, {"a": 1, "b": "x"})
    PredictionLogger._append_jsonl(path, {"a": 2, "b": "y"})

    lines = path.read_text().splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1, "b": "x"}
    assert json.loads(lines[1]) == {"a": 2, "b": "y"}
