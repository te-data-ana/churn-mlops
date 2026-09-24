import json
from datetime import UTC
from pathlib import Path

from churn_mlops.serving.prediction_logger import PredictionLogger
from churn_mlops.serving.predictor import PredictionResult
from churn_mlops.serving.schemas import InputFeatures


def test_log_prediction_writes_prediction_event(
    tmp_path: Path,
    sample_input: InputFeatures,
    sample_prediction_result: PredictionResult,
) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    error_path = tmp_path / "prediction_errors.jsonl"

    logger = PredictionLogger(
        prediction_log_path=pred_path,
        error_log_path=error_path,
        log_features=True,
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
    assert record["features"] == sample_input.model_dump()


def test_log_error_writes_error_event_with_features(
    tmp_path: Path,
    sample_input: InputFeatures,
) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    error_path = tmp_path / "prediction_errors.jsonl"

    logger = PredictionLogger(
        prediction_log_path=pred_path,
        error_log_path=error_path,
        log_features=True,
    )

    exc = ValueError("invalid input")

    logger.log_error(
        request_id="req-1",
        latency_ms=12.3,
        exception=exc,
        traceback_text="traceback content",
        features=sample_input,
    )

    lines = error_path.read_text().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["event"] == "prediction_error"
    assert record["request_id"] == "req-1"
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


def test_timestamp_returns_utc_datetime() -> None:
    ts = PredictionLogger._timestamp()

    assert ts.tzinfo == UTC
