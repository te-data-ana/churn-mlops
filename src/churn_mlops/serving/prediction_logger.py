# from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from churn_mlops.serving.predictor import PredictionResult, Predictor
from churn_mlops.serving.schemas import PredictionErrorEvent, PredictionEvent
from churn_mlops.use_cases import UseCase


class PredictionLogger:
    """Persist structured prediction and error events as JSONL."""

    def __init__(
        self,
        prediction_log_path: Path,
        error_log_path: Path,
        log_features: bool,
        use_case: UseCase,
    ) -> None:
        self.prediction_log_path = prediction_log_path
        self.error_log_path = error_log_path
        self.log_features = log_features
        self.use_case = use_case

        self.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        *,
        request_id: str,
        latency_ms: float,
        prediction: PredictionResult,
        features: BaseModel | Mapping[str, Any] | None,
    ) -> None:

        flag_features = self.log_features and features is not None

        event = PredictionEvent(
            request_id=request_id,
            timestamp=prediction.timestamp,
            latency_ms=latency_ms,
            model_name=prediction.metadata.model_name,
            model_alias=prediction.metadata.model_alias,
            model_version=prediction.metadata.model_version,
            predicted_class=prediction.predicted_class,
            predicted_probability=prediction.predicted_probability,
            threshold=prediction.metadata.model_threshold,
            features=self._feature_mapping(features) if flag_features else None,
        )

        self._append_jsonl(
            self.prediction_log_path,
            self._event_record(event),
        )

    def log_error(
        self,
        *,
        request_id: str,
        latency_ms: float,
        predictor: Predictor,
        exception: Exception,
        traceback_text: str,
        features: BaseModel | Mapping[str, Any] | None = None,
    ) -> None:

        flag_features = self.log_features and features is not None

        event = PredictionErrorEvent(
            request_id=request_id,
            timestamp=predictor._timestamp(),
            latency_ms=latency_ms,
            model_name=predictor.metadata.model_name,
            model_alias=predictor.metadata.model_alias,
            model_version=predictor.metadata.model_version,
            error_type=type(exception).__name__,
            error_message=str(exception),
            traceback=traceback_text,
            features=self._feature_mapping(features) if flag_features else None,
        )

        self._append_jsonl(
            self.error_log_path,
            self._event_record(event),
        )

    def _event_record(
        self, event: PredictionEvent | PredictionErrorEvent
    ) -> dict[str, Any]:
        record = event.model_dump(mode="json", exclude_none=True)
        record[self.use_case.time_column] = record.pop("timestamp")
        return record

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    @staticmethod
    def _feature_mapping(
        features: BaseModel | Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if features is None:
            return None
        if isinstance(features, BaseModel):
            return features.model_dump(mode="json")
        return dict(features)
