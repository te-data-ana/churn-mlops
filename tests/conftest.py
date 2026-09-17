from unittest.mock import Mock

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_mlops import TrainingResult
from churn_mlops.config.schemas import (
    ClassifierConfig,
    DataConfig,
    EvaluationConfig,
    FeatureBuilderConfig,
    PreprocessingConfig,
    RegistryConfig,
    TrainingConfig,
)
from churn_mlops.models.features import FeatureBuilder
from churn_mlops.tracking.registry import ModelRegistry


@pytest.fixture
def registry():
    registry = ModelRegistry()
    registry.client = Mock()
    return registry


@pytest.fixture
def mock_taining_result():
    return TrainingResult(
        trained_pipeline=Pipeline(steps=[("features", FeatureBuilder())]),
        metrics={},
        classifier_config={},
        metadata={
            "train_rows": 10,
            "test_rows": 5,
            "feature_count": 3,
            "feature_names_in": ["a", "b"],
            "feature_names_out": ["a", "b", "c"],
        },
    )


@pytest.fixture
def config_factory():
    def _create(**overrides):

        config = TrainingConfig(
            data=DataConfig(
                target_column="churn",
                test_size=0.5,
                random_state=42,
            ),
            feature_builder=FeatureBuilderConfig(
                feature_params={
                    "engagement_window": 30,
                    "inactive_threshold": 15,
                    "late_payer_threshold": 15,
                }
            ),
            preprocessing=PreprocessingConfig(
                numeric_impute_strategy="mean",
                categorical_impute_strategy="most_frequent",
            ),
            model=ClassifierConfig(
                classifier="lr",
                classifier_params={
                    "random_state": 42,
                },
            ),
            evaluation=EvaluationConfig(
                threshold=0.5,
            ),
            registry=RegistryConfig(register_model=False, registry_params={}),
        )

        for key, value in overrides.items():
            setattr(config.model, key, value)

        return config

    return _create


@pytest.fixture(scope="session")
def sample_training_df():
    df = pd.DataFrame(
        {
            "tenure": [0, 12, 24, 60, None, 50, 42, 14, 6, None],
            "contract_length": [
                "Monthly",
                "Quarterly",
                "Annual",
                "Monthly",
                "Monthly",
                "Quarterly",
                "Annual",
                "Monthly",
                "Monthly",
                None,
            ],
            "usage_frequency": [10, 20, 30, None, 15, 5, 25, 10, None, 5],
            "last_interaction": [5, 15, 20, 10, None, 25, 5, 2, 1, None],
            "total_spend": [0, 400, 1000, None, 750, 150, 450, 800, None, 250],
            "support_calls": [10, 6, 1, 2, None, 9, 7, 3, 4, 5],
            "payment_delay": [25, 10, 0, None, 30, 22, 12, 2, 0, None],
            "churn": [1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
        }
    )
    df["contract_length"] = df["contract_length"].astype("str")
    return df


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def sample_training_data(sample_df):
    X = sample_df.copy()
    y = pd.Series([1, 0, 0, 1, 0], name="churn")

    return X, y
