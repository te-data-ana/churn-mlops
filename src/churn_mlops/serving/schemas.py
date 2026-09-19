from typing import Literal

from pydantic import BaseModel, Field


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
