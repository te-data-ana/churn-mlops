from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InputFeatures(BaseModel):
    age: int = Field(ge=18, le=115)
    tenure: int = Field(ge=0)
    usage_frequency: int = Field(ge=0)
    support_calls: int = Field(ge=0)
    payment_delay: int = Field(ge=0)
    last_interaction: int = Field(ge=0)
    total_spend: float = Field(ge=0)

    gender: Literal["Female", "Male"]

    subscription_type: Literal[
        "Basic",
        "Standard",
        "Premium",
    ]

    contract_length: Literal[
        "Monthly",
        "Quarterly",
        "Annual",
    ]


class BaseInferenceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    timestamp_utc: datetime
    latency_ms: float = Field(ge=0)

    # Optional feature logging
    features: InputFeatures | None = None


class PredictionEvent(BaseInferenceEvent):
    event: Literal["prediction"] = "prediction"

    model_name: str
    model_alias: str
    model_version: int

    predicted_class: Literal[0, 1]
    predicted_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


class PredictionErrorEvent(BaseInferenceEvent):
    event: Literal["prediction_error"] = "prediction_error"

    error_type: str
    error_message: str
    traceback: str
