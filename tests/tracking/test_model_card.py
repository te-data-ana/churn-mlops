import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.tracking import (
    ModelCardBuilder,
    ModelCardContext,
    log_model_card,
)


@pytest.mark.unit
def test_build_model_card_from_training_result(
    registered_model: dict[str, object],
    config_factory: TrainingConfig,
) -> None:
    result = registered_model["result"]
    config = config_factory(classifier="lr")

    context = ModelCardContext(
        model_name="test-model",
        model_version=1,
        promotion_decision=None,
    )

    markdown = ModelCardBuilder().build(
        result=result,
        config=config,
        context=context,
    )

    assert "# Model Card" in markdown
    assert "## Model Metadata" in markdown
    assert "## Evaluation on Test Set" in markdown
    assert str(context.model_version) in markdown
    for feature in result.metadata["feature_names_out"]:
        assert f"- {feature}" in markdown
    assert f"| Classifier Alias | {config.model.classifier} |" in markdown
    assert f"| Classifier Name | {result.classifier_config['model_name']} |" in markdown
    assert re.search(r"\| ROC AUC \| \d\.\d{4} \|", markdown)
    assert re.search(r"\| Accuracy \| \d\.\d{4} \|", markdown)


@pytest.mark.unit
def test_build_includes_promotion_section(
    registered_model: dict[str, object],
    config_factory: TrainingConfig,
) -> None:
    result = registered_model["result"]
    config = config_factory()

    decision = SimpleNamespace(
        candidate_metric=0.9123,
        champion_metric=0.9000,
        required_delta=0.005,
        metric_delta=0.0123,
        promote=True,
        reason="Candidate beats champion.",
    )

    context = ModelCardContext(
        model_name="test-model",
        model_version=2,
        promotion_decision=decision,
    )

    markdown = ModelCardBuilder().build(
        result=result,
        config=config,
        context=context,
    )

    assert "## Promotion Decision" in markdown
    assert "| Promoted | True |" in markdown
    assert "Candidate beats champion." in markdown


@pytest.mark.unit
def test_build_renders_na_for_missing_champion_values(
    registered_model: dict[str, object],
    config_factory: TrainingConfig,
) -> None:
    result = registered_model["result"]
    config = config_factory()

    decision = SimpleNamespace(
        candidate_metric=0.91,
        champion_metric=None,
        required_delta=0.005,
        metric_delta=None,
        promote=False,
        reason="No champion model available.",
    )

    context = ModelCardContext(
        model_name="test-model",
        model_version=1,
        promotion_decision=decision,
    )

    markdown = ModelCardBuilder().build(
        result=result,
        config=config,
        context=context,
    )

    assert "| Champion ROC AUC | N/A |" in markdown
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
