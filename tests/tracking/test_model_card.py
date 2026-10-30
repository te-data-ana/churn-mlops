import re
from typing import Any
from unittest.mock import patch

import pytest

from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.tracking import (
    ModelCardBuilder,
    log_model_card,
)
from churn_mlops.tracking.manifest import TrainingManifest


@pytest.mark.unit
def test_build_model_card_from_training_manifest(
    config_factory: TrainingConfig,
    training_manifest_factory: TrainingManifest,
) -> None:
    config = config_factory(classifier="lr")

    context = training_manifest_factory()

    markdown = ModelCardBuilder().build(
        config=config,
        context=context,
    )

    assert "# Model Card" in markdown
    assert "## Model Metadata" in markdown
    assert "## Evaluation on Out-of-Time Test Set" in markdown
    assert str(context.model_version) in markdown
    for feature in context.feature_names_out:
        assert f"- {feature}" in markdown
    assert f"| Classifier Alias | {context.classifier_alias} |" in markdown
    assert f"| Classifier Name | {context.classifier_name} |" in markdown
    assert "| Calibration Method | none |" in markdown
    assert "| Threshold Selection Strategy | max_f1 |" in markdown
    assert "| Threshold Selection Data | out_of_time_test |" in markdown
    assert "maximizing F1 on the out-of-time test set" in markdown
    assert f"{context.threshold:.6f}" in markdown
    assert re.search(r"\| ROC AUC \| \d\.\d{4} \|", markdown)
    assert re.search(r"\| Calibration Error \(ECE\) \| \d\.\d{4} \|", markdown)
    assert re.search(r"\| Accuracy \| \d\.\d{4} \|", markdown)


@pytest.mark.unit
def test_build_includes_promotion_section(
    config_factory: TrainingConfig,
    training_manifest_factory: TrainingManifest,
) -> None:
    config = config_factory()

    decision = {
        "candidate_score": 0.9123,
        "champion_score": 0.9000,
        "score_delta": 0.0123,
        "min_delta": 0.005,
        "promote": True,
        "reason": "Candidate beats champion.",
    }

    context = training_manifest_factory(decision)

    markdown = ModelCardBuilder().build(
        config=config,
        context=context,
    )

    assert "## Promotion Decision" in markdown
    assert "| Promoted | True |" in markdown
    assert "Candidate beats champion." in markdown


@pytest.mark.unit
def test_build_renders_na_for_missing_champion_values(
    config_factory: TrainingConfig,
    training_manifest_factory: TrainingManifest,
) -> None:
    config = config_factory()

    decision = {
        "candidate_score": 0.91,
        "champion_score": None,
        "score_delta": None,
        "min_delta": 0.005,
        "promote": True,
        "reason": "No champion model available.",
    }

    context = training_manifest_factory(decision)

    markdown = ModelCardBuilder().build(
        config=config,
        context=context,
    )

    assert "| Weighted Champion Score | N/A |" in markdown
    assert "| Actual Delta | N/A |" in markdown


@pytest.mark.unit
@patch("churn_mlops.tracking.model_card.mlflow.log_artifact")
def test_log_model_card_calls_mlflow(
    mock_log_artifact: Any,
) -> None:
    log_model_card(
        markdown="# Model Card",
        artifact_path="documentation",
    )

    mock_log_artifact.assert_called_once()

    _, kwargs = mock_log_artifact.call_args
    assert kwargs["artifact_path"] == "documentation"
