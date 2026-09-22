"""Fixtures shared by configuration, model, and training tests."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from churn_mlops.config.schemas import (
    ClassifierConfig,
    DataConfig,
    EvaluationConfig,
    FeatureBuilderConfig,
    PreprocessingConfig,
    RegistryConfig,
    TrainingConfig,
)


@pytest.fixture
def config_factory() -> Callable[..., TrainingConfig]:
    def _create(**overrides) -> TrainingConfig:

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


@pytest.fixture
def sample_training_df() -> pd.DataFrame:
    # Includes the churn target for training and end-to-end pipeline tests.
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


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    # Feature and preprocessing tests use this frame without the churn target.
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
def sample_training_data(
    sample_features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    X = sample_features_df.copy()
    y = pd.Series([1, 0, 0, 1, 0], name="churn")

    return X, y


@pytest.fixture
def generated_training_df() -> pd.DataFrame:
    """Return deterministic schema-valid training data for integration tests."""
    return pd.DataFrame(
        {
            "Age": [22, 31, 44, 57, 29, 63, 38, 48, 71, 26, 52, 35],
            "Tenure": [2, 8, 14, 30, 4, 42, 18, 25, 55, 6, 36, 11],
            "Usage Frequency": [3, 8, 12, 22, 5, 28, 14, 18, 30, 4, 20, 9],
            "Support Calls": [8, 5, 3, 1, 7, 0, 4, 2, 0, 9, 1, 6],
            "Payment Delay": [20, 12, 5, 0, 18, 0, 7, 2, 0, 25, 1, 15],
            "Last Interaction": [35, 20, 12, 3, 28, 1, 10, 6, 2, 40, 4, 18],
            "total_spend": [
                100.0,
                450.0,
                900.0,
                240.0,
                250.0,
                400.0,
                900.0,
                800.0,
                650.0,
                150.0,
                300.0,
                700.0,
            ],
            "Gender": [
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
            ],
            "Subscription Type": [
                "Basic",
                "Standard",
                "Premium",
                "Premium",
                "Basic",
                "Premium",
                "Standard",
                "Premium",
                "Premium",
                "Basic",
                "Standard",
                "Standard",
            ],
            "Contract Length": [
                "Monthly",
                "Quarterly",
                "Annual",
                "Annual",
                "Monthly",
                "Annual",
                "Quarterly",
                "Annual",
                "Annual",
                "Monthly",
                "Quarterly",
                "Monthly",
            ],
            "Churn": [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        },
        index=pd.Index(range(1001, 1013), name="customerid"),
    )


@pytest.fixture
def generated_training_csv(tmp_path: Path, generated_training_df: pd.DataFrame) -> Path:
    """Write generated training data to a temporary CSV file."""
    path = tmp_path / "training.csv"
    generated_training_df.to_csv(path)
    return path


@pytest.fixture
def generated_inference_csv(
    tmp_path: Path, generated_training_df: pd.DataFrame
) -> Path:
    """Write generated inference data to a temporary CSV file."""
    path = tmp_path / "inference.csv"
    generated_training_df.drop(columns=["Churn"]).to_csv(path)
    return path


@pytest.fixture
def mlflow_test_setup(tmp_path: Path) -> dict[str, Any]:
    """Return a dictionary of MLflow test parameters for use in training and inference tests."""
    return {
        "tmp_path": tmp_path,
        "model_name": "test-model",
        "model_alias": "test-champion",
        "experiment_name": "test-experiment",
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
    }


@pytest.fixture
def sample_config_yaml(mlflow_test_setup: dict[str, Any]) -> Path:
    """Write a sample training configuration to a temporary YAML file."""

    tmp_path = mlflow_test_setup["tmp_path"]
    model_name = mlflow_test_setup["model_name"]
    model_alias = mlflow_test_setup["model_alias"]

    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "target_column": "churn",
                    "test_size": 0.25,
                    "random_state": 42,
                },
                "feature_builder": {
                    "feature_params": {
                        "engagement_window": 30,
                        "inactive_threshold": 15,
                        "late_payer_threshold": 15,
                    }
                },
                "preprocessing": {
                    "numeric_impute_strategy": "median",
                    "categorical_impute_strategy": "most_frequent",
                },
                "model": {
                    "classifier": "lr",
                    "classifier_params": {"random_state": 42},
                },
                "evaluation": {"threshold": 0.5},
                "registry": {
                    "register_model": True,
                    "registry_params": {
                        "model_name": model_name,
                        "alias": model_alias,
                        "promotion_delta": 0.005,
                    },
                },
            }
        )
    )
    return config_path


@pytest.fixture
def registered_model(
    mlflow_test_setup: dict[str, Any],
    sample_config_yaml: Path,
    generated_training_csv: Path,
) -> dict[str, object]:
    """Train and register a model in an isolated temporary MLflow store."""
    from churn_mlops.training import run_training_job

    tmp_path = mlflow_test_setup["tmp_path"]
    model_name = mlflow_test_setup["model_name"]
    model_alias = mlflow_test_setup["model_alias"]
    experiment_name = mlflow_test_setup["experiment_name"]
    tracking_uri = mlflow_test_setup["tracking_uri"]
    artifact_dir = tmp_path / "artifacts"

    result = run_training_job(
        config_file=sample_config_yaml.name,
        config_dir=tmp_path,
        training_file=generated_training_csv.name,
        index_col="customerid",
        data_dir=tmp_path,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        artifact_dir=artifact_dir,
    )

    return {
        "result": result,
        "tracking_uri": tracking_uri,
        "artifact_dir": artifact_dir,
        "model_name": model_name,
        "model_alias": model_alias,
    }
