from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sklearn.linear_model import LogisticRegression

from churn_mlops.serving import model_loader


@pytest.mark.unit
def test_load_model_returns_model_with_registry_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock MLflow, settings, and registry calls to isolate metadata assembly.
    model = Mock()
    registry = Mock()
    registry.load_model_by_alias.return_value = model
    registry.get_model_version_by_alias.return_value = SimpleNamespace(version=7)
    registry.get_threshold_by_alias.return_value = 0.6

    settings = SimpleNamespace(
        model_name="churn-risk",
        model_alias="champion",
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
    assert loaded_model.metadata.threshold == 0.6
    registry.load_model_by_alias.assert_called_once_with(
        model_name="churn-risk",
        alias="champion",
    )


@pytest.mark.unit
def test_load_model_uses_exact_registered_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock()
    registry = Mock()
    registry.load_model_by_version.return_value = model
    registry.get_threshold_by_version.return_value = 0.4
    settings = SimpleNamespace(
        model_name="churn-risk",
        model_alias="champion",
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )
    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())

    loaded_model = model_loader.load_model(model_version=9)

    assert loaded_model.model is model
    assert loaded_model.metadata.model_version == 9
    assert loaded_model.metadata.model_alias is None
    assert loaded_model.metadata.threshold == 0.4
    registry.load_model_by_version.assert_called_once_with(
        model_name="churn-risk",
        version=9,
    )


@pytest.mark.unit
def test_load_model_uses_logged_model_uri_without_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Mock()
    load_mlflow_model = Mock(return_value=model)
    settings = SimpleNamespace(
        model_name="churn-risk",
        model_alias="champion",
        mlflow_tracking_uri="sqlite:///mlflow.db",
    )
    registry = Mock()
    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())
    monkeypatch.setattr(model_loader.mlflow.sklearn, "load_model", load_mlflow_model)

    loaded_model = model_loader.load_model(
        model_uri="runs:/run-123/model",
        model_threshold=0.37,
    )

    assert loaded_model.model is model
    assert loaded_model.metadata.model_version is None
    assert loaded_model.metadata.model_uri == "runs:/run-123/model"
    assert loaded_model.metadata.model_alias is None
    assert loaded_model.metadata.threshold == 0.37
    load_mlflow_model.assert_called_once_with("runs:/run-123/model")
    registry.load_model_by_alias.assert_not_called()
    registry.load_model_by_version.assert_not_called()


@pytest.mark.integration
def test_load_model_loads_real_logged_mlflow_model_uri(
    mlflow_test_setup: dict[str, object],
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

    loaded_model = model_loader.load_model(
        tracking_uri=tracking_uri,
        model_uri=model_info.model_uri,
        model_threshold=0.5,
    )

    assert loaded_model.metadata.model_version is None
    assert loaded_model.metadata.model_uri == model_info.model_uri
    assert loaded_model.model.predict_proba([[1]])[0, 1] < 0.5
