from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sklearn.linear_model import LogisticRegression

from churn_mlops.serving import model_loader
from churn_mlops.tracking.manifest import LoggedModelReference, TrainingManifest


@pytest.mark.unit
def test_load_model_returns_model_with_registry_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock MLflow, settings, and registry calls to isolate metadata assembly.
    model = Mock()
    registry = Mock()
    registry.load_registered_model.return_value = SimpleNamespace(
        model=model,
        model_version=SimpleNamespace(version="7"),
        model_threshold=0.6,
    )

    settings = SimpleNamespace(
        model_name="churn-risk",
        model_alias="champion",
        model_version=None,
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )

    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())

    loaded_model = model_loader.load_model()

    assert loaded_model.model is model
    assert loaded_model.metadata.model_name == "churn-risk"
    assert loaded_model.metadata.model_alias == "champion"
    assert loaded_model.metadata.model_version == 7
    assert loaded_model.metadata.model_threshold == 0.6
    registry.load_registered_model.assert_called_once_with(
        model_name="churn-risk",
        model_alias="champion",
        model_version=None,
    )


@pytest.mark.unit
def test_load_model_uses_exact_registered_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock()
    registry = Mock()
    registry.load_registered_model.return_value = SimpleNamespace(
        model=model,
        model_version=SimpleNamespace(version="9"),
        model_threshold=0.4,
    )
    settings = SimpleNamespace(
        model_name="churn-risk",
        model_alias="champion",
        model_version=None,
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )
    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())

    loaded_model = model_loader.load_model(model_version=9)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_version == 9
    assert loaded_model.metadata.model_alias is None
    assert loaded_model.metadata.model_threshold == 0.4
    registry.load_registered_model.assert_called_once_with(
        model_name="churn-risk",
        model_alias=None,
        model_version=9,
    )


@pytest.mark.unit
def test_load_model_uses_manifest_threshold_for_registered_version(
    monkeypatch: pytest.MonkeyPatch,
    training_manifest_factory: Callable[..., TrainingManifest],
) -> None:
    model = Mock()
    registry = Mock()
    registry.load_model_by_version.return_value = model
    settings = SimpleNamespace(
        model_name="configured-model",
        model_alias="champion",
        model_version=None,
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )
    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())

    manifest = training_manifest_factory()
    loaded_model = model_loader.load_model(manifest=manifest)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_name == manifest.model_name
    assert loaded_model.metadata.model_version == manifest.model_version
    assert loaded_model.metadata.model_threshold == manifest.threshold
    assert loaded_model.metadata.model_reference == manifest.model_reference
    registry.load_model_by_version.assert_called_once_with(
        model_name=manifest.model_name,
        model_version=manifest.model_version,
    )
    registry.load_registered_model.assert_not_called()


@pytest.mark.unit
def test_load_model_uses_logged_model_reference_without_registry(
    monkeypatch: pytest.MonkeyPatch,
    training_manifest_factory: Callable[..., TrainingManifest],
) -> None:
    model = Mock()
    load_mlflow_model = Mock(return_value=model)
    settings = SimpleNamespace(
        model_name="churn-risk",
        model_alias="champion",
        model_version=None,
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )
    registry = Mock()
    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())
    monkeypatch.setattr(model_loader.mlflow.sklearn, "load_model", load_mlflow_model)

    manifest = replace(
        training_manifest_factory(),
        model_version=None,
        threshold=0.37,
    )
    loaded_model = model_loader.load_model(manifest=manifest)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_version is None
    assert loaded_model.metadata.model_reference == manifest.model_reference
    assert loaded_model.metadata.model_alias is None
    assert loaded_model.metadata.model_threshold == 0.37
    load_mlflow_model.assert_called_once_with("runs:/run-123/model")
    registry.load_registered_model.assert_not_called()


@pytest.mark.integration
def test_load_model_loads_logged_mlflow_model_using_manifest(
    mlflow_test_setup: dict[str, object],
    training_manifest_factory: Callable[..., TrainingManifest],
) -> None:
    tracking_uri = str(mlflow_test_setup["tracking_uri"])
    model_loader.mlflow.set_tracking_uri(tracking_uri)
    model_loader.mlflow.set_experiment("unregistered-model-test")
    estimator = LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1])

    with model_loader.mlflow.start_run():
        model_info = model_loader.mlflow.sklearn.log_model(
            estimator,
            name="model",
        )

    manifest = replace(
        training_manifest_factory(),
        model_version=None,
        model_reference=LoggedModelReference.from_model_info(model_info),
    )
    loaded_model = model_loader.load_model(
        tracking_uri=tracking_uri,
        manifest=manifest,
    )

    assert loaded_model.metadata.model_version is None
    assert loaded_model.metadata.model_reference == manifest.model_reference
    assert loaded_model.model.predict_proba([[1]])[0, 1] < 0.5
