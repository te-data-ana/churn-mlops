from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from churn_mlops.use_cases.churn.serving import ChurnInputFeatures


def test_churn_input_features_accepts_boundary_values() -> None:
    input_features = ChurnInputFeatures(
        age=115,
        tenure=0,
        usage_frequency=0,
        support_calls=0,
        payment_delay=0,
        last_interaction=0,
        total_spend=0.0,
        gender="Male",
        subscription_type="Basic",
        contract_length="Monthly",
    )

    assert input_features.age == 115
    assert input_features.total_spend == 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", 17),
        ("age", 116),
        ("support_calls", -1),
        ("payment_delay", -1),
        ("gender", "XY"),
        ("subscription_type", "Free"),
        ("contract_length", "undefined"),
    ],
)
def test_churn_input_features_rejects_invalid_values(
    sample_churn_input_factory: Callable[..., dict],
    field: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        ChurnInputFeatures(**sample_churn_input_factory(**{field: value}))
