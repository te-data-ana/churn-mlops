"""Manifest metadata linking training runs to downstream scoring stages."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingManifest:
    """Run metadata needed to identify an immutable model for later scoring."""

    run_id: str
    run_name: str
    model_name: str
    model_version: int
    target_column: str
    classifier_name: str
    classifier_alias: str
    split_name: str
    split_metadata: dict[str, str | int]
    feature_count: int
    feature_names_in: list[str]
    feature_names_out: list[str]
    threshold: float
    metrics: dict[str, float]
    promotion_decision: dict

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
            "run_id": self.run_id,
            "run_name": self.run_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "target_column": self.target_column,
            "classifier_name": self.classifier_name,
            "classifier_alias": self.classifier_alias,
            "split_name": self.split_name,
            "split_metadata": self.split_metadata,
            "feature_count": self.feature_count,
            "feature_names_in": self.feature_names_in,
            "feature_names_out": self.feature_names_out,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "promotion_decision": self.promotion_decision,
        }

    @classmethod
    def read(cls, path: Path) -> "TrainingManifest":
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
            "run_id",
            "run_name",
            "model_name",
            "model_version",
            "target_column",
            "classifier_name",
            "classifier_alias",
            "split_name",
            "split_metadata",
            "feature_count",
            "feature_names_in",
            "feature_names_out",
            "threshold",
            "metrics",
            "promotion_decision",
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

        if not isinstance(payload["model_version"], int):
            raise TypeError(f"Training manifest '{path}' has an invalid model_version.")

        if not isinstance(payload["metrics"], dict) or not isinstance(
            payload["split_metadata"], dict
        ):
            raise TypeError(f"Training manifest '{path}' has invalid metadata maps.")

        if not all(
            isinstance(payload[key], str)
            for key in (
                "run_id",
                "model_name",
                "target_column",
                "classifier_name",
                "classifier_alias",
                "split_name",
                "run_name",
            )
        ):
            raise TypeError(f"Training manifest '{path}' has invalid text fields.")

        if not isinstance(payload["feature_count"], int):
            raise TypeError(f"Training manifest '{path}' has an invalid feature count.")

        if not isinstance(payload["feature_names_in"], list):
            raise TypeError(f"Training manifest '{path}' has invalid input features.")

        if not isinstance(payload["feature_names_out"], list):
            raise TypeError(f"Training manifest '{path}' has invalid output features.")

        if not isinstance(payload["threshold"], float):
            raise TypeError(f"Training manifest '{path}' has an invalid threshold.")

        try:
            metrics = {key: float(value) for key, value in payload["metrics"].items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Training manifest '{path}' contains invalid metric values."
            ) from exc

        if not all(
            isinstance(value, (str, int)) and not isinstance(value, bool)
            for value in payload["split_metadata"].values()
        ):
            raise TypeError(
                f"Training manifest '{path}' contains invalid split metadata."
            )

        if not isinstance(payload["promotion_decision"], dict):
            raise TypeError(
                f"Training manifest '{path}' has an invalid promotion decision."
            )

        return cls(
            run_id=payload["run_id"],
            run_name=payload["run_name"],
            model_name=payload["model_name"],
            model_version=payload["model_version"],
            target_column=payload["target_column"],
            classifier_name=payload["classifier_name"],
            classifier_alias=payload["classifier_alias"],
            split_name=payload["split_name"],
            split_metadata=payload["split_metadata"],
            feature_count=payload["feature_count"],
            feature_names_in=payload["feature_names_in"],
            feature_names_out=payload["feature_names_out"],
            threshold=payload["threshold"],
            metrics=metrics,
            promotion_decision=payload["promotion_decision"],
        )
