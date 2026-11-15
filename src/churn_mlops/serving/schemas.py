from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


def prediction_response_schema(time_column: str) -> type[BaseModel]:
    """Create the API response schema for a use case's timestamp field."""
    return create_model(
        f"PredictionResponse_{time_column}",
        **{
            time_column: (datetime, ...),
            "predicted_class": (Literal[0, 1], ...),
            "predicted_probability": (float, Field(ge=0.0, le=1.0)),
            "metadata": (dict[str, Any], ...),
        },
    )


class BaseInferenceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    timestamp: datetime
    latency_ms: float = Field(ge=0)

    model_name: str
    model_alias: str | None
    model_version: int

    # Optional feature logging
    features: dict[str, Any] | None = None


class PredictionEvent(BaseInferenceEvent):
    event: Literal["prediction"] = "prediction"

    predicted_class: Literal[0, 1]
    predicted_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


class PredictionErrorEvent(BaseInferenceEvent):
    event: Literal["prediction_error"] = "prediction_error"

    error_type: str
    error_message: str
    traceback: str
