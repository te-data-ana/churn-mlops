import pytest
from pydantic import ValidationError

from churn_mlops.serving.schemas import InputFeatures


@pytest.mark.unit
def test_input_features_rejects_invalid_values() -> None:
    # Multiple invalid fields exercise the input validation boundary.
    with pytest.raises(ValidationError):
        InputFeatures(
            age=0,
            tenure=0,
            usage_frequency=0,
            support_calls=-1,
            payment_delay=-1,
            last_interaction=0,
            total_spend=0.0,
            gender="XY",
            subscription_type="Free",
            contract_length="undefined",
        )
