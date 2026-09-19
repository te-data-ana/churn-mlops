from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from churn_mlops.serving import model_loader


@pytest.mark.unit
def test_load_model_returns_model_with_registry_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Mock MLflow, settings, and registry calls to isolate metadata assembly.
    model = Mock()
    registry = Mock()
    registry.load_model.return_value = model
    registry.get_model_version_by_alias.return_value = SimpleNamespace(version=7)
    registry.get_threshold_by_alias.return_value = 0.6

    settings = SimpleNamespace(model_name="churn-propensity", model_alias="champion")

    monkeypatch.setattr(model_loader, "ModelRegistry", lambda: registry)
    monkeypatch.setattr(model_loader, "ServingSettings", lambda: settings)
    monkeypatch.setattr(model_loader.mlflow, "set_tracking_uri", Mock())

    loaded_model = model_loader.load_model()

    assert loaded_model.model is model
    assert loaded_model.metadata.model_name == "churn-propensity"
    assert loaded_model.metadata.model_alias == "champion"
    assert loaded_model.metadata.model_version == 7
    assert loaded_model.metadata.threshold == 0.6
    registry.load_model.assert_called_once_with(
        model_name="churn-propensity",
        alias="champion",
    )
