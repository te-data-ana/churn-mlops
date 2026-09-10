import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    df = pd.DataFrame(
        {
            "tenure": [0, 12, 24, 60, None],
            "contract_length": ["Monthly", "Quarterly", "Annual", "Monthly", None],
            "usage_frequency": [10, 20, 30, None, 5],
            "last_interaction": [5, 15, 20, 10, None],
            "total_spend": [0, 400, 1000, None, 750],
            "support_calls": [10, 6, 1, 2, None],
            "payment_delay": [25, 10, 0, None, 30],
        }
    )
    df["contract_length"] = df["contract_length"].astype("str")
    return df


@pytest.fixture
def sample_training_data(sample_df):
    X = sample_df.copy()
    y = pd.Series([1, 0, 0, 1, 0], name="churn")

    return X, y
