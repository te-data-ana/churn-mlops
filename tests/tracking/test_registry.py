from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException

from churn_mlops.tracking.registry import ModelRegistry


@pytest.mark.unit
def test_registry_set_alias_calls_mlflow_client(registry: ModelRegistry) -> None:
    registry.set_alias(
        model_name="my_model",
        alias="champion",
        version=5,
    )

    registry.client.set_registered_model_alias.assert_called_once_with(
        name="my_model",
        alias="champion",
        version=5,
    )


@pytest.mark.unit
def test_registry_get_model_version(registry: ModelRegistry) -> None:
    registry.get_model_version(
        model_name="my_model",
        version=3,
    )

    registry.client.get_model_version.assert_called_once_with(
        name="my_model",
        version="3",
    )


@pytest.mark.unit
def test_registry_update_model_description(registry: ModelRegistry) -> None:
    model_version = Mock()
    registry.client.update_model_version.return_value = model_version

    result = registry.update_model_description(
        model_name="my_model",
        version=3,
        description="Updated model card.",
    )

    registry.client.update_model_version.assert_called_once_with(
        name="my_model",
        version="3",
        description="Updated model card.",
    )

    assert result is model_version


@pytest.mark.unit
def test_registry_get_metric_by_alias(registry: ModelRegistry) -> None:
    version = Mock()
    version.run_id = "abc"

    run = Mock()
    run.data.metrics = {"roc_auc": 0.84}

    registry.get_model_version_by_alias = Mock(return_value=version)
    registry.client.get_run.return_value = run

    metric = registry.get_metric_by_alias(
        model_name="my_model",
        alias="champion",
        metric_name="roc_auc",
    )

    assert metric == 0.84


@pytest.mark.unit
def test_registry_get_threshold_by_alias(registry: ModelRegistry) -> None:
    version = Mock()
    version.run_id = "abc"

    run = Mock()
    run.data.params = {"threshold": 0.4}

    registry.get_model_version_by_alias = Mock(return_value=version)
    registry.client.get_run.return_value = run

    threshold = registry.get_threshold_by_alias(
        model_name="my_model",
        alias="champion",
    )

    assert threshold == 0.4


@pytest.mark.unit
def test_registry_returns_none_when_champion_alias_is_missing(
    registry: ModelRegistry,
) -> None:
    # Simulate MLflow reporting that no champion alias exists.
    registry.get_model_version_by_alias = Mock(
        side_effect=MlflowException("no model registered with alias='champion'")
    )

    champion = registry.get_champion_version("my_model")

    assert champion is None


@pytest.mark.unit
def test_registry_returns_champion_version_when_alias_exists(
    registry: ModelRegistry,
) -> None:
    champion = Mock()
    registry.get_model_version_by_alias = Mock(return_value=champion)

    result = registry.get_champion_version("my_model")

    assert result is champion
