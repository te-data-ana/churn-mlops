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
def test_training_manifest_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"run_id": "run-123"}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing fields"):
        TrainingManifest.read(path)
