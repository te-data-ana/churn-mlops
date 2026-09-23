from pathlib import Path

import pytest

from churn_mlops.config import RuntimeSettings


@pytest.mark.unit
def test_runtime_settings_read_container_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tracking_dir = tmp_path / "tracking"
    artifact_dir = tmp_path / "artifacts"

    monkeypatch.setenv("TRACKING_DIR", str(tracking_dir))
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.setenv("API_PORT", "9123")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "container-training")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    settings = RuntimeSettings()

    assert settings.tracking_dir == tracking_dir
    assert settings.artifact_dir == artifact_dir
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 9123
    assert settings.mlflow_experiment_name == "container-training"
    assert settings.mlflow_tracking_uri == f"sqlite:///{tracking_dir / 'mlflow.db'}"


@pytest.mark.unit
def test_explicit_mlflow_uri_overrides_derived_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'custom.db'}"
    monkeypatch.setenv("TRACKING_DIR", str(tmp_path / "tracking"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    settings = RuntimeSettings()

    assert settings.mlflow_tracking_uri == tracking_uri


@pytest.mark.unit
def test_runtime_settings_use_default_mlflow_experiment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)

    settings = RuntimeSettings()

    assert settings.mlflow_experiment_name == "test"
