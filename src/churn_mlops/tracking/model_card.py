import tempfile
from dataclasses import dataclass
from textwrap import dedent

import mlflow

from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.tracking.promotion import PromotionDecision
from churn_mlops.training import TrainingResult


@dataclass(frozen=True)
class ModelCardContext:
    model_name: str
    model_version: int
    promotion_decision: PromotionDecision | None


class ModelCardBuilder:
    def build(
        self,
        *,
        result: TrainingResult,
        config: TrainingConfig,
        context: ModelCardContext,
    ) -> str:
        """Build a Markdown model card for a trained model.

        The generated model card summarizes model metadata, dataset
        characteristics, feature information, preprocessing configuration,
        evaluation metrics, and promotion decision details when available.

        Args:
            result: Training output containing evaluation metrics, dataset
                metadata, and classifier configuration information.
            config: Training configuration used to build and evaluate the
                model, including preprocessing, model, data, and evaluation
                settings.
            context: Additional model card context containing the registered
                model name, model version, and optional promotion decision.

        Returns:
            A Markdown-formatted string representing the model card.

        Raises:
            KeyError: If required keys are missing from
                ``result.metrics``, ``result.metadata``, or
                ``result.classifier_config``.
            AttributeError: If any required attributes are missing from
                ``result``, ``config``, ``context``, or the promotion
                decision object.
        """

        metrics = result.metrics
        metadata = result.metadata
        decision = context.promotion_decision

        features = "\n".join(f"- {f}" for f in metadata["feature_names_out"])

        promotion_section = ""
        if decision:
            champion_metric = (
                f"{decision.champion_metric:.4f}"
                if decision.champion_metric is not None
                else "N/A"
            )
            metric_delta = (
                f"{decision.metric_delta:.4f}"
                if decision.metric_delta is not None
                else "N/A"
            )

            promotion_section = dedent(
                f"""
                ## Promotion Decision

                | Property | Value |
                |----------|-------|
                | Candidate ROC AUC | {decision.candidate_metric:.4f} |
                | Champion ROC AUC | {champion_metric} |
                | Required Delta | {decision.required_delta:.4f} |
                | Actual Delta | {metric_delta} |
                | Promoted | {decision.promote} |

                Reason: {decision.reason}
                """
            ).strip()

        return f"""
# Model Card

## Model Metadata

| Property | Value |
|----------|-------|
| Model Name | {context.model_name} |
| Version | {context.model_version} |
| Classifier Alias | {config.model.classifier} |
| Classifier Name | {result.classifier_config["model_name"]} |

## Intended Use

Predict customer {config.data.target_column} probability.

## Training Dataset

Trained on historical {config.data.target_column} data.

| Property | Value |
|----------|-------|
| Train Rows | {metadata["train_rows"]} |
| Test Rows | {metadata["test_rows"]} |
| Feature Count | {metadata["feature_count"]} |

## Model Features

{features}

## Imputation Strategy

| Feature Type | Imputation Strategy |
|--------------|---------------------|
| Numeric | {config.preprocessing.numeric_impute_strategy} |
| Categorical | {config.preprocessing.categorical_impute_strategy} |

## Evaluation on Test Set

| Metric | Value |
|--------|-------|
| ROC AUC | {metrics["roc_auc"]:.4f} |
| PR AUC | {metrics["pr_auc"]:.4f} |
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 | {metrics["f1"]:.4f} |
| Brier Score | {metrics["brier_score"]:.4f} |

## Threshold

Applied probability threshold: {config.evaluation.threshold}

{promotion_section}
"""


def log_model_card(
    markdown: str,
    artifact_path: str = "documentation",
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".md",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(markdown)

    mlflow.log_artifact(
        f.name,
        artifact_path=artifact_path,
    )
