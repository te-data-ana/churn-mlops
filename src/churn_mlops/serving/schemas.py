from typing import Literal

from pydantic import BaseModel


class InputFeatures(BaseModel):
    age: int
    tenure: int
    usage_frequency: int
    support_calls: int
    payment_delay: int
    last_interaction: int
    total_spend: float

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
