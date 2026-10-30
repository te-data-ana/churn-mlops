from dataclasses import dataclass
from math import isfinite

from churn_mlops.tracking.registry import ModelRegistry

_LOWER_IS_BETTER = {"brier_score", "calibration_error"}


@dataclass(frozen=True)
class PromotionDecision:
    """Candidate comparison and promotion outcome details."""

    candidate_score: float
    champion_score: float | None
    score_delta: float | None
    min_delta: float
    promote: bool
    reason: str


class PromotionService:
    """Calculate configured quality scores and update champion aliases."""

    @staticmethod
    def calculate_weighted_score(
        metrics: dict[str, float],
        metric_weights: dict[str, float],
    ) -> float:
        """Calculate a normalized quality score from configured metrics.

        Brier score and calibration error are inverted so that higher scores
        always indicate better model quality.

        Args:
            metrics: Evaluation metrics for a model.
            metric_weights: Non-negative weights for configured metrics.

        Returns:
            Weighted score in [0, 1].

        Raises:
            ValueError: If metrics or weights are missing or invalid.
        """
        if not metric_weights or any(
            not isfinite(weight) or weight < 0 for weight in metric_weights.values()
        ):
            raise ValueError(
                "Promotion metric weights must be finite, non-negative, and nonempty."
            )
        total_weight = sum(metric_weights.values())
        if total_weight <= 0:
            raise ValueError("Promotion metric weights must have a positive total.")
        missing_metrics = set(metric_weights).difference(metrics)
        if missing_metrics:
            raise ValueError(
                "Missing metrics required for promotion score: "
                f"{', '.join(sorted(missing_metrics))}."
            )

        score = 0.0
        for metric_name, weight in metric_weights.items():
            value = metrics[metric_name]
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"Promotion metric '{metric_name}' must be between 0 and 1."
                )
            oriented_value = 1 - value if metric_name in _LOWER_IS_BETTER else value
            score += weight * oriented_value
        return score / total_weight

    def evaluate_weighted_candidate(
        self,
        *,
        candidate_metrics: dict[str, float],
        champion_metrics: dict[str, float] | None,
        metric_weights: dict[str, float],
        min_delta: float,
    ) -> PromotionDecision:
        """Compare candidate and champion using a configured weighted score.

        Args:
            candidate_metrics: Candidate model evaluation metrics.
            champion_metrics: Champion metrics, or ``None`` when no champion
                exists.
            metric_weights: Metrics and weights used for both model scores.
            min_delta: Strict score improvement required to promote.

        Returns:
            A promotion decision with weighted scores and metric details.

        Raises:
            ValueError: If candidate metrics or configuration are invalid.
        """
        candidate_score = self.calculate_weighted_score(
            metrics=candidate_metrics,
            metric_weights=metric_weights,
        )
        if champion_metrics is None:
            return PromotionDecision(
                candidate_score=candidate_score,
                champion_score=None,
                score_delta=None,
                min_delta=min_delta,
                promote=True,
                reason="No champion model existed yet.",
            )

        missing_champion_metrics = set(metric_weights).difference(champion_metrics)
        if missing_champion_metrics:
            missing = ", ".join(sorted(missing_champion_metrics))
            return PromotionDecision(
                candidate_score=candidate_score,
                champion_score=None,
                score_delta=None,
                min_delta=min_delta,
                promote=False,
                reason=(
                    "Champion cannot be compared because its logged run is "
                    f"missing required metrics: {missing}."
                ),
            )

        champion_score = self.calculate_weighted_score(
            metrics=champion_metrics,
            metric_weights=metric_weights,
        )
        score_delta = candidate_score - champion_score
        promote = score_delta > min_delta
        reason = (
            "Candidate weighted quality score improved beyond the required delta."
            if promote
            else "Candidate does not achieve the required weighted quality improvement."
        )
        return PromotionDecision(
            candidate_score=candidate_score,
            champion_score=champion_score,
            score_delta=score_delta,
            min_delta=min_delta,
            promote=promote,
            reason=reason,
        )

    def promote_candidate(
        self,
        decision: PromotionDecision,
        registry: ModelRegistry,
        model_name: str,
        candidate_version: int,
    ) -> None:
        """Promote an approved candidate and preserve the former champion.

        Args:
            decision: Previously evaluated promotion decision.
            registry: Model registry used to update aliases.
            model_name: Registered model name.
            candidate_version: Version to assign the ``champion`` alias.

        Raises:
            ValueError: If the decision does not approve promotion.
        """

        if not decision.promote:
            raise ValueError(
                "A candidate can only be promoted, if it was evaluated and the evaluation was positive."
            )

        if decision.champion_score is not None:
            # Preserve the previous champion for rollback before replacing it.
            current_champion = registry.get_champion_version(model_name=model_name)
            registry.set_alias(
                model_name=model_name,
                alias="former_champion",
                version=current_champion.version,
            )
        # promote new champion
        registry.set_alias(
            model_name=model_name,
            alias="champion",
            version=candidate_version,
        )
