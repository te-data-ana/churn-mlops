from unittest.mock import Mock

import pytest
from mlflow.exceptions import MlflowException

from churn_mlops.tracking.registry import ModelRegistry


@pytest.mark.unit
def test_registry_registers_model_info_uri(
    registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_info = Mock(model_uri="models:/logged-model")
    registered_version = Mock()
    register_model = Mock(return_value=registered_version)
    monkeypatch.setattr(
        "churn_mlops.tracking.registry.mlflow.register_model",
        register_model,
    )

    version = registry.register_model(model_info=model_info, model_name="my_model")

    assert version is registered_version
    register_model.assert_called_once_with(
        model_uri="models:/logged-model",
        name="my_model",
    )


@pytest.mark.unit
def test_registry_set_alias_calls_mlflow_client(registry: ModelRegistry) -> None:
    registry.set_alias(
        model_name="my_model",
        model_alias="champion",
        model_version=5,
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
        model_version=3,
    )
    registry.client.get_model_version.assert_called_once_with(
        name="my_model",
        version="3",
    )


@pytest.mark.unit
def test_registry_loads_an_exact_model_version(
    registry: ModelRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_model = Mock(return_value=object())
    monkeypatch.setattr(
        "churn_mlops.tracking.registry.mlflow.sklearn.load_model", load_model
    )

    registry.load_model_by_version(model_name="my_model", model_version=7)

    load_model.assert_called_once_with("models:/my_model/7")


@pytest.mark.unit
def test_registry_loads_registered_alias_with_its_threshold(
    registry: ModelRegistry,
) -> None:
    model_version = Mock(version="7", run_id="run-7")
    model = object()
    registry.get_model_version_by_alias = Mock(return_value=model_version)
    registry.load_model_by_version = Mock(return_value=model)
    registry.client.get_run.return_value.data.params = {"threshold": "0.35"}

    loaded = registry.load_registered_model(
        model_name="my_model",
        model_alias="champion",
    )

    assert loaded.model is model
    assert loaded.model_version is model_version
    assert loaded.model_threshold == 0.35
    registry.get_model_version_by_alias.assert_called_once_with(
        model_name="my_model",
        model_alias="champion",
    )
    registry.load_model_by_version.assert_called_once_with(
        model_name="my_model",
        model_version=7,
    )
    registry.client.get_run.assert_called_once_with(run_id="run-7")


@pytest.mark.unit
def test_registry_loads_registered_version_with_its_threshold(
    registry: ModelRegistry,
) -> None:
    model_version_object = Mock(version="7", run_id="run-7")
    model = object()
    registry.get_model_version = Mock(return_value=model_version_object)
    registry.load_model_by_version = Mock(return_value=model)
    registry.client.get_run.return_value.data.params = {"threshold": "0.35"}

    loaded = registry.load_registered_model(model_name="my_model", model_version=7)

    assert loaded.model is model
    assert loaded.model_version is model_version_object
    assert loaded.model_threshold == 0.35
    registry.get_model_version.assert_called_once_with(
        model_name="my_model",
        model_version=7,
    )


@pytest.mark.unit
def test_registry_requires_exactly_one_version_selector(
    registry: ModelRegistry,
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        registry.load_registered_model(model_name="my_model")


@pytest.mark.unit
def test_registry_update_model_description(registry: ModelRegistry) -> None:
    model_version = Mock()
    registry.client.update_model_version.return_value = model_version

    result = registry.update_model_description(
        model_name="my_model",
        model_version="3",
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
        model_alias="champion",
        metric_name="roc_auc",
    )

    assert metric == 0.84


@pytest.mark.unit
def test_registry_get_metrics_by_alias(registry: ModelRegistry) -> None:
    version = Mock()
    version.run_id = "abc"
    run = Mock()
    run.data.metrics = {"roc_auc": 0.84, "calibration_error": 0.09}
    registry.get_model_version_by_alias = Mock(return_value=version)
    registry.client.get_run.return_value = run

    metrics = registry.get_metrics_by_alias(
        model_name="my_model",
        model_alias="champion",
    )

    assert metrics == {"roc_auc": 0.84, "calibration_error": 0.09}


@pytest.mark.unit
def test_registry_returns_none_when_champion_alias_is_missing(
    registry: ModelRegistry,
) -> None:
    # Simulate MLflow reporting that no champion alias exists.
    registry.get_model_version_by_alias = Mock(
        side_effect=MlflowException("no model registered with alias='champion'")
    )

    champion = registry.get_champion_version(model_name="my_model")

    assert champion is None


@pytest.mark.unit
def test_registry_returns_champion_version_when_alias_exists(
    registry: ModelRegistry,
) -> None:
    champion = Mock()
    registry.get_model_version_by_alias = Mock(return_value=champion)

    result = registry.get_champion_version(model_name="my_model")

    assert result is champion
