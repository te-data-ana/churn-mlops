from collections.abc import Callable

import pytest


@pytest.fixture
def sample_churn_input_factory() -> Callable[..., dict]:
    def factory(**kwargs) -> dict:
        data = {
            "age": 45,
            "tenure": 24,
            "usage_frequency": 12,
            "support_calls": 1,
            "payment_delay": 0,
            "last_interaction": 5,
            "total_spend": 800.0,
            "gender": "Female",
            "subscription_type": "Premium",
            "contract_length": "Annual",
        }
        data.update(kwargs)

        return data

    return factory
