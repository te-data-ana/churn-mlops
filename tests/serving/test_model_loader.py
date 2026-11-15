from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sklearn.linear_model import LogisticRegression

from churn_mlops.serving import model_loader
from churn_mlops.tracking.manifest import LoggedModelReference, TrainingManifest
from churn_mlops.use_cases import get_use_case

CHURN_USE_CASE = get_use_case("churn")


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

    loaded_model = model_loader.load_model(use_case=CHURN_USE_CASE)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_name == "churn-risk"
    assert loaded_model.metadata.model_alias == "champion"
    assert loaded_model.metadata.model_version == 7
    assert loaded_model.metadata.model_threshold == 0.6
    assert loaded_model.use_case is CHURN_USE_CASE
    registry.load_registered_model.assert_called_once_with(
        model_name="churn-risk",
        expected_use_case="churn",
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

    loaded_model = model_loader.load_model(use_case=CHURN_USE_CASE, model_version=9)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_version == 9
    assert loaded_model.metadata.model_alias is None
    assert loaded_model.metadata.model_threshold == 0.4
    registry.load_registered_model.assert_called_once_with(
        model_name="churn-risk",
        expected_use_case="churn",
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
    registry.load_registered_model.return_value = SimpleNamespace(model=model)
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
    loaded_model = model_loader.load_model(use_case=CHURN_USE_CASE, manifest=manifest)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_name == manifest.model.model_name
    assert loaded_model.metadata.model_version == manifest.model.model_version
    assert loaded_model.metadata.model_threshold == manifest.training.threshold
    assert loaded_model.metadata.model_reference == manifest.model.model_reference
    registry.load_registered_model.assert_called_once_with(
        model_name=manifest.model.model_name,
        expected_use_case="churn",
        model_version=manifest.model.model_version,
    )


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
    registry.validate_run_use_case = Mock()

    manifest = training_manifest_factory()
    manifest = replace(
        manifest,
        model=replace(manifest.model, model_version=None),
        training=replace(manifest.training, threshold=0.37),
    )
    loaded_model = model_loader.load_model(use_case=CHURN_USE_CASE, manifest=manifest)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_version is None
    assert loaded_model.metadata.model_reference == manifest.model.model_reference
    assert loaded_model.metadata.model_alias is None
    assert loaded_model.metadata.model_threshold == 0.37
    load_mlflow_model.assert_called_once_with("runs:/run-123/model")
    registry.validate_run_use_case.assert_called_once_with(
        run_id=manifest.run.run_id,
        expected_use_case="churn",
    )
    registry.load_registered_model.assert_not_called()


@pytest.mark.unit
def test_load_model_rejects_manifest_for_another_use_case(
    monkeypatch: pytest.MonkeyPatch,
    training_manifest_factory: Callable[..., TrainingManifest],
) -> None:
    manifest = training_manifest_factory()
    manifest = replace(
        manifest,
        model=replace(manifest.model, use_case="alternate"),
    )
    registry_factory = Mock()
    monkeypatch.setattr(model_loader, "ModelRegistry", registry_factory)

    with pytest.raises(ValueError, match="selects use case 'alternate'"):
        model_loader.load_model(use_case=CHURN_USE_CASE, manifest=manifest)

    registry_factory.assert_not_called()


@pytest.mark.integration
def test_load_model_loads_logged_mlflow_model_using_manifest(
    mlflow_test_setup: dict[str, object],
    training_manifest_factory: Callable[..., TrainingManifest],
    tmp_path: Path,
) -> None:
    tracking_uri = str(mlflow_test_setup["tracking_uri"])
    model_loader.mlflow.set_tracking_uri(tracking_uri)
    experiment_id = model_loader.mlflow.create_experiment(
        "unregistered-model-test",
        artifact_location=str(tmp_path / "artifacts"),
    )
    model_loader.mlflow.set_experiment(experiment_id=experiment_id)
    estimator = LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1])

    with model_loader.mlflow.start_run(experiment_id=experiment_id) as run:
        model_loader.mlflow.log_param("use_case", "churn")
        model_info = model_loader.mlflow.sklearn.log_model(
            estimator,
            name="model",
        )
        run_id = run.info.run_id

    manifest = training_manifest_factory()
    manifest = replace(
        manifest,
        run=replace(manifest.run, run_id=run_id),
        model=replace(
            manifest.model,
            model_version=None,
            model_reference=LoggedModelReference.from_model_info(model_info),
        ),
    )
    loaded_model = model_loader.load_model(
        use_case=CHURN_USE_CASE,
        tracking_uri=tracking_uri,
        manifest=manifest,
    )

    assert loaded_model.metadata.model_version is None
    assert loaded_model.metadata.model_reference == manifest.model.model_reference
    assert loaded_model.model.predict_proba([[1]])[0, 1] < 0.5
