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
    manifest = training_manifest_factory()
    output = tmp_path / "manifests" / "run.json"

    manifest.write(output)

    assert TrainingManifest.read(output) == manifest


@pytest.mark.unit
def test_unregistered_training_manifest_round_trips_logged_model_reference(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    manifest = replace(training_manifest_factory(), model_version=None)
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
    payload["model_version"] = None
    payload["model_reference"] = None
    path = tmp_path / "invalid-model-reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TypeError, match="model_reference"):
        TrainingManifest.read(path)


@pytest.mark.unit
def test_training_manifest_rejects_incomplete_logged_model_reference(
    tmp_path: Path,
    training_manifest_factory: TrainingManifest,
) -> None:
    payload = training_manifest_factory().to_dict()
    payload["model_reference"].pop("model_uri")
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
    del payload["calibration"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration"):
        TrainingManifest.read(path)
