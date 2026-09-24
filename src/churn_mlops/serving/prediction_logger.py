# from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from churn_mlops.serving.predictor import PredictionResult
from churn_mlops.serving.schemas import (
    InputFeatures,
    PredictionErrorEvent,
    PredictionEvent,
)


class PredictionLogger:
    """Persist structured prediction and error events as JSONL."""

    def __init__(
        self,
        prediction_log_path: Path,
        error_log_path: Path,
        log_features: bool,
    ) -> None:
        self.prediction_log_path = prediction_log_path
        self.error_log_path = error_log_path
        self.log_features = log_features

        self.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        *,
        request_id: str,
        latency_ms: float,
        prediction: PredictionResult,
        features: InputFeatures | None,
    ) -> None:

        flag_features = self.log_features and features is not None

        event = PredictionEvent(
            request_id=request_id,
            timestamp_utc=self._timestamp(),
            latency_ms=latency_ms,
            model_name=prediction.metadata.model_name,
            model_alias=prediction.metadata.model_alias,
            model_version=prediction.metadata.model_version,
            predicted_class=prediction.predicted_class,
            predicted_probability=prediction.predicted_probability,
            threshold=prediction.metadata.threshold,
            features=features if flag_features else None,
        )

        self._append_jsonl(
            self.prediction_log_path,
            event.model_dump(
                mode="json",
                exclude_none=True,
            ),
        )

    def log_error(
        self,
        *,
        request_id: str,
        latency_ms: float,
        exception: Exception,
        traceback_text: str,
        features: InputFeatures | None = None,
    ) -> None:

        flag_features = self.log_features and features is not None

        event = PredictionErrorEvent(
            request_id=request_id,
            timestamp_utc=self._timestamp(),
            latency_ms=latency_ms,
            error_type=type(exception).__name__,
            error_message=str(exception),
            traceback=traceback_text,
            features=features if flag_features else None,
        )

        self._append_jsonl(
            self.error_log_path,
            event.model_dump(
                mode="json",
                exclude_none=True,
            ),
        )

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    @staticmethod
    def _timestamp() -> datetime:
        return datetime.now(UTC)
