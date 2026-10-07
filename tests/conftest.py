"""Fixtures shared by configuration, model, and training tests."""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
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
from churn_mlops.evaluation import metrics
from churn_mlops.tracking.manifest import TrainingManifest


@pytest.fixture
def training_manifest_factory() -> Callable[..., TrainingManifest]:
    def _create(promotion_decision: dict | None = None) -> TrainingManifest:

        evaluation = metrics.evaluate_model(
            y_true=np.array([0, 0, 1, 1]),
            y_prob=np.array([0.1, 0.4, 0.6, 0.9]),
            fit_time_sec=1.2,
            pred_time_sec=0.3,
        )

        return TrainingManifest(
            run_id="run-123",
            run_name="rf__oot_26q1",
            model_name="churn-propensity",
            model_version=4,
            target_column="churn",
            classifier_name="clf_name",
            classifier_alias="clf_alias",
            split_name="oot_26q1",
            split_metadata={
                "train_rows": 100,
                "test_rows": 25,
                "train_start": "2025-01-01T00:00:00",
                "train_end": "2025-12-31T00:00:00",
                "test_start": "2026-01-01T00:00:00",
                "test_end": "2026-03-31T00:00:00",
            },
            feature_count=2,
            feature_names_in=["feature_in_1", "feature_in_2"],
            feature_names_out=["feature_out_1", "feature_out_2"],
            threshold=0.5,
            metrics=vars(evaluation),
            promotion_decision=promotion_decision or {},
        )

    return _create


@pytest.fixture
def config_factory() -> Callable[..., TrainingConfig]:
    def _create(**overrides) -> TrainingConfig:

        config = TrainingConfig(
            data=DataConfig(
                target_column="churn",
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
    monkeypatch: pytest.MonkeyPatch,
    generated_training_df: pd.DataFrame,
    mlflow_test_setup: dict[str, Any],
    sample_config_yaml: Path,
) -> dict[str, object]:
    """Train and register a model in an isolated temporary MLflow store."""
    from churn_mlops import train
    from churn_mlops.data import splitting, storage

    tmp_path = mlflow_test_setup["tmp_path"]
    model_name = mlflow_test_setup["model_name"]
    model_alias = mlflow_test_setup["model_alias"]
    experiment_name = mlflow_test_setup["experiment_name"]
    tracking_uri = mlflow_test_setup["tracking_uri"]
    artifact_dir = tmp_path / "artifacts"

    settings = SimpleNamespace(
        config_dir=tmp_path,
        data_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment_name=experiment_name,
    )
    monkeypatch.setattr(storage, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(splitting, "RuntimeSettings", lambda: settings)
    monkeypatch.setattr(train, "RuntimeSettings", lambda: settings)

    split_name = "integration"

    df = generated_training_df.copy().reset_index(drop=True)
    df["reference_date"] = pd.date_range("2026-01-01", periods=len(df))

    storage.write_partitioned_dataset(
        df=df,
        dataset_name="integration",
        timestamp_column="reference_date",
    )

    _ = splitting.create_time_based_split(
        dataset_name="integration",
        timestamp_column="reference_date",
        start_date="2026-01-01",
        split_date="2026-01-06",
        end_date="2026-01-10",
        split_name=split_name,
    )

    result = train.run_training_job(
        config_file=sample_config_yaml.name,
        split_name=split_name,
        config_dir=tmp_path,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        artifact_dir=artifact_dir,
        register_model=True,
        promote_model=True,
    )

    return {
        "result": result,
        "tracking_uri": tracking_uri,
        "artifact_dir": artifact_dir,
        "model_name": model_name,
        "model_alias": model_alias,
    }
