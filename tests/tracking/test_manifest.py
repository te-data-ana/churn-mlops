import json
from dataclasses import replace
from pathlib import Path

import pytest

from churn_mlops.tracking.manifest import TrainingManifest


@pytest.mark.unit
def test_training_manifest_round_trips_json(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    manifest = training_manifest_factory({"promote": False})
    manifest = replace(
        manifest,
        training=replace(manifest.training, tuning_enabled=True),
    )
    output = tmp_path / "manifests" / "run.json"

    manifest.write(output)

    assert TrainingManifest.read(output) == manifest
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert list(payload) == [
        "schema_version",
        "run",
        "model",
        "data",
        "training",
        "evaluation",
        "lifecycle",
    ]
    assert list(payload["training"]) == [
        "classifier",
        "features",
        "tuning",
        "calibration",
        "threshold",
    ]
    assert list(payload["lifecycle"]) == ["registration", "promotion"]
    assert payload["schema_version"] == 1
    assert payload["data"]["time_column"] == "reference_date"
    assert payload["training"]["tuning"]["enabled"] is True
    assert payload["lifecycle"]["registration"]["enabled"] is True
    assert payload["lifecycle"]["promotion"]["enabled"] is True


@pytest.mark.unit
def test_unregistered_training_manifest_round_trips_logged_model_reference(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    manifest = training_manifest_factory()
    manifest = replace(
        manifest,
        model=replace(manifest.model, model_version=None),
    )
    output = tmp_path / "manifest.json"

    manifest.write(output)

    assert TrainingManifest.read(output) == manifest


@pytest.mark.unit
def test_training_manifest_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"run_id": "run-123"}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        TrainingManifest.read(path)


@pytest.mark.unit
def test_training_manifest_requires_model_version_or_reference(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    payload = training_manifest_factory().to_dict()
    payload["model"]["version"] = None
    payload["model"]["reference"] = None
    path = tmp_path / "invalid-model-reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TypeError, match="reference"):
        TrainingManifest.read(path)


@pytest.mark.unit
def test_training_manifest_rejects_incomplete_logged_model_reference(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    payload = training_manifest_factory().to_dict()
    payload["model"]["reference"].pop("model_uri")
    path = tmp_path / "invalid-model-info.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="model_uri"):
        TrainingManifest.read(path)


@pytest.mark.unit
def test_training_manifest_requires_current_metadata_fields(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    payload = training_manifest_factory().to_dict()
    del payload["training"]["calibration"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration"):
        TrainingManifest.read(path)


@pytest.mark.unit
def test_training_manifest_rejects_unsupported_schema_version(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    payload = training_manifest_factory().to_dict()
    payload["schema_version"] = 0
    path = tmp_path / "unsupported-version.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported schema_version"):
        TrainingManifest.read(path)
