"""Manifest metadata linking training runs to downstream scoring stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo


@dataclass(frozen=True)
class LoggedModelReference:
    """Portable identity and URI for a logged MLflow model artifact."""

    model_uri: str
    model_uuid: str
    artifact_path: str
    mlflow_version: str

    @classmethod
    def from_model_info(cls, model_info: ModelInfo) -> LoggedModelReference:
        """Create a JSON-safe reference from MLflow's logged model metadata.

        Args:
            model_info: Metadata returned when MLflow logs the model.

        Returns:
            A portable reference containing the model URI and identity fields.
        """
        return cls(
            model_uri=model_info.model_uri,
            model_uuid=model_info.model_uuid,
            artifact_path=model_info.artifact_path,
            mlflow_version=model_info.mlflow_version,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LoggedModelReference:
        """Validate and construct a reference from a manifest object.

        Args:
            payload: JSON object containing logged-model identity fields.

        Returns:
            Validated logged-model reference.

        Raises:
            TypeError: If any identity field is not non-empty text.
            ValueError: If a required identity field is missing.
        """
        required = {"model_uri", "model_uuid", "artifact_path", "mlflow_version"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(
                "Training manifest model_reference is missing fields: "
                f"{', '.join(sorted(missing))}."
            )
        if not all(isinstance(payload[key], str) and payload[key] for key in required):
            raise TypeError("Training manifest has invalid model_reference fields.")
        return cls(
            model_uri=payload["model_uri"],
            model_uuid=payload["model_uuid"],
            artifact_path=payload["artifact_path"],
            mlflow_version=payload["mlflow_version"],
        )

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-safe model reference fields."""
        return {
            "model_uri": self.model_uri,
            "model_uuid": self.model_uuid,
            "artifact_path": self.artifact_path,
            "mlflow_version": self.mlflow_version,
        }


@dataclass(frozen=True)
class ManifestRun:
    """Identity of the MLflow training run."""

    run_id: str
    run_name: str


@dataclass(frozen=True)
class ManifestModel:
    """Model identity and logged artifact reference."""

    model_name: str
    model_version: int | None
    model_reference: LoggedModelReference


@dataclass(frozen=True)
class ManifestData:
    """Target, time, and split information used for training."""

    target_column: str
    time_column: str
    split_name: str
    split_metadata: dict[str, str | int]


@dataclass(frozen=True)
class ManifestTraining:
    """Resolved training configuration and generated feature metadata."""

    classifier_name: str
    classifier_alias: str
    feature_count: int
    feature_names_in: list[str]
    feature_names_out: list[str]
    tuning_enabled: bool
    calibration: str
    threshold_strategy: str
    threshold_data: str
    threshold: float


@dataclass(frozen=True)
class ManifestEvaluation:
    """Evaluation results for the trained model."""

    metrics: dict[str, float]


@dataclass(frozen=True)
class ManifestLifecycle:
    """Resolved model registration and promotion actions."""

    registration_enabled: bool
    promotion_enabled: bool
    promotion_decision: dict


@dataclass(frozen=True)
class TrainingManifest:
    """Run metadata needed to identify an immutable model for later scoring."""

    run: ManifestRun
    model: ManifestModel
    data: ManifestData
    training: ManifestTraining
    evaluation: ManifestEvaluation
    lifecycle: ManifestLifecycle
    schema_version: int = 1

    def write(self, path: Path) -> None:
        """Write the manifest as formatted JSON.

        Args:
            path: Destination JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation.

        Returns:
            Manifest fields as a JSON-compatible dictionary.
        """
        return {
            "schema_version": self.schema_version,
            "run": {
                "id": self.run.run_id,
                "name": self.run.run_name,
            },
            "model": {
                "name": self.model.model_name,
                "version": self.model.model_version,
                "reference": self.model.model_reference.to_dict(),
            },
            "data": {
                "target_column": self.data.target_column,
                "time_column": self.data.time_column,
                "split": {
                    "name": self.data.split_name,
                    "metadata": self.data.split_metadata,
                },
            },
            "training": {
                "classifier": {
                    "name": self.training.classifier_name,
                    "alias": self.training.classifier_alias,
                },
                "features": {
                    "count": self.training.feature_count,
                    "names_in": self.training.feature_names_in,
                    "names_out": self.training.feature_names_out,
                },
                "tuning": {
                    "enabled": self.training.tuning_enabled,
                },
                "calibration": {
                    "method": self.training.calibration,
                },
                "threshold": {
                    "strategy": self.training.threshold_strategy,
                    "selected_on": self.training.threshold_data,
                    "value": self.training.threshold,
                },
            },
            "evaluation": {
                "metrics": self.evaluation.metrics,
            },
            "lifecycle": {
                "registration": {
                    "enabled": self.lifecycle.registration_enabled,
                },
                "promotion": {
                    "enabled": self.lifecycle.promotion_enabled,
                    "decision": self.lifecycle.promotion_decision,
                },
            },
        }

    @classmethod
    def read(cls, path: Path) -> TrainingManifest:
        """Read and validate a training manifest from JSON.

        Args:
            path: Source JSON file.

        Returns:
            Validated training manifest.

        Raises:
            TypeError: If the manifest contains values of invalid types.
            ValueError: If required fields or metric values are invalid.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "run",
            "model",
            "data",
            "training",
            "evaluation",
            "lifecycle",
        }
        if not isinstance(payload, dict):
            raise TypeError(
                f"Training manifest '{path}' is invalid; expected a JSON object."
            )

        missing = required.difference(payload)
        if missing:
            raise ValueError(
                f"Training manifest '{path}' is missing fields: "
                f"{', '.join(sorted(missing))}."
            )

        if type(payload["schema_version"]) is not int:
            raise TypeError(
                f"Training manifest '{path}' has an invalid schema_version."
            )
        if payload["schema_version"] != 1:
            raise ValueError(
                f"Training manifest '{path}' uses unsupported schema_version "
                f"{payload['schema_version']}."
            )

        run = _require_object(payload, "run", path)
        model = _require_object(payload, "model", path)
        data = _require_object(payload, "data", path)
        training = _require_object(payload, "training", path)
        evaluation = _require_object(payload, "evaluation", path)
        lifecycle = _require_object(payload, "lifecycle", path)

        _require_fields(run, {"id", "name"}, "run", path)
        _require_fields(model, {"name", "version", "reference"}, "model", path)
        _require_fields(
            data,
            {"target_column", "time_column", "split"},
            "data",
            path,
        )
        _require_fields(
            training,
            {"classifier", "features", "tuning", "calibration", "threshold"},
            "training",
            path,
        )
        _require_fields(evaluation, {"metrics"}, "evaluation", path)
        _require_fields(lifecycle, {"registration", "promotion"}, "lifecycle", path)

        split = _require_object(data, "split", path)
        classifier = _require_object(training, "classifier", path)
        features = _require_object(training, "features", path)
        tuning = _require_object(training, "tuning", path)
        calibration = _require_object(training, "calibration", path)
        threshold = _require_object(training, "threshold", path)
        registration = _require_object(lifecycle, "registration", path)
        promotion = _require_object(lifecycle, "promotion", path)

        _require_fields(split, {"name", "metadata"}, "data.split", path)
        _require_fields(classifier, {"name", "alias"}, "training.classifier", path)
        _require_fields(
            features,
            {"count", "names_in", "names_out"},
            "training.features",
            path,
        )
        _require_fields(tuning, {"enabled"}, "training.tuning", path)
        _require_fields(calibration, {"method"}, "training.calibration", path)
        _require_fields(
            threshold,
            {"strategy", "selected_on", "value"},
            "training.threshold",
            path,
        )
        _require_fields(registration, {"enabled"}, "lifecycle.registration", path)
        _require_fields(promotion, {"enabled", "decision"}, "lifecycle.promotion", path)

        model_version = model["version"]
        if model_version is not None and type(model_version) is not int:
            raise TypeError(f"Training manifest '{path}' has an invalid model.version.")
        if not isinstance(model["reference"], dict):
            raise TypeError(f"Training manifest '{path}' has invalid model.reference.")
        model_reference = LoggedModelReference.from_dict(model["reference"])

        if not isinstance(evaluation["metrics"], dict) or not isinstance(
            split["metadata"], dict
        ):
            raise TypeError(f"Training manifest '{path}' has invalid metadata maps.")

        text_fields = (
            (run, "id"),
            (run, "name"),
            (model, "name"),
            (data, "target_column"),
            (data, "time_column"),
            (split, "name"),
            (classifier, "name"),
            (classifier, "alias"),
            (calibration, "method"),
            (threshold, "strategy"),
            (threshold, "selected_on"),
        )
        if not all(isinstance(obj[key], str) for obj, key in text_fields):
            raise TypeError(f"Training manifest '{path}' has invalid text fields.")

        if type(features["count"]) is not int:
            raise TypeError(
                f"Training manifest '{path}' has invalid training.features.count."
            )

        if not isinstance(features["names_in"], list) or not all(
            isinstance(name, str) for name in features["names_in"]
        ):
            raise TypeError(f"Training manifest '{path}' has invalid input features.")

        if not isinstance(features["names_out"], list) or not all(
            isinstance(name, str) for name in features["names_out"]
        ):
            raise TypeError(f"Training manifest '{path}' has invalid output features.")

        if type(tuning["enabled"]) is not bool:
            raise TypeError(
                f"Training manifest '{path}' has invalid training.tuning.enabled."
            )
        if type(registration["enabled"]) is not bool:
            raise TypeError(
                f"Training manifest '{path}' has invalid lifecycle.registration.enabled."
            )
        if type(promotion["enabled"]) is not bool:
            raise TypeError(
                f"Training manifest '{path}' has invalid lifecycle.promotion.enabled."
            )

        if not isinstance(threshold["value"], float):
            raise TypeError(
                f"Training manifest '{path}' has an invalid threshold value."
            )

        try:
            metrics = {
                key: float(value) for key, value in evaluation["metrics"].items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Training manifest '{path}' contains invalid metric values."
            ) from exc

        if not all(
            isinstance(value, (str, int)) and not isinstance(value, bool)
            for value in split["metadata"].values()
        ):
            raise TypeError(
                f"Training manifest '{path}' contains invalid split metadata."
            )

        if not isinstance(promotion["decision"], dict):
            raise TypeError(
                f"Training manifest '{path}' has an invalid lifecycle.promotion.decision."
            )

        return cls(
            schema_version=payload["schema_version"],
            run=ManifestRun(run_id=run["id"], run_name=run["name"]),
            model=ManifestModel(
                model_name=model["name"],
                model_version=model_version,
                model_reference=model_reference,
            ),
            data=ManifestData(
                target_column=data["target_column"],
                time_column=data["time_column"],
                split_name=split["name"],
                split_metadata=split["metadata"],
            ),
            training=ManifestTraining(
                classifier_name=classifier["name"],
                classifier_alias=classifier["alias"],
                feature_count=features["count"],
                feature_names_in=features["names_in"],
                feature_names_out=features["names_out"],
                tuning_enabled=tuning["enabled"],
                calibration=calibration["method"],
                threshold_strategy=threshold["strategy"],
                threshold_data=threshold["selected_on"],
                threshold=threshold["value"],
            ),
            evaluation=ManifestEvaluation(metrics=metrics),
            lifecycle=ManifestLifecycle(
                registration_enabled=registration["enabled"],
                promotion_enabled=promotion["enabled"],
                promotion_decision=promotion["decision"],
            ),
        )


def _require_object(
    payload: dict[str, Any],
    key: str,
    path: Path,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(
            f"Training manifest '{path}' has an invalid object in section '{key}'."
        )
    return value


def _require_fields(
    payload: dict[str, Any],
    required: set[str],
    section: str,
    path: Path,
) -> None:
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"Training manifest '{path}' section '{section}' is missing fields: "
            f"{', '.join(sorted(missing))}."
        )
