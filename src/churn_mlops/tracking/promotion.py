from dataclasses import dataclass

from churn_mlops.tracking.registry import ModelRegistry


@dataclass(frozen=True)
class PromotionDecision:
    candidate_metric: float
    champion_metric: float | None
    metric_delta: float | None
    required_delta: float
    reason: str

    @property
    def promote(self) -> bool:
        if self.champion_metric is None:
            return True

        assert self.metric_delta is not None

        return self.metric_delta > self.required_delta


class PromotionService:
    def evaluate_candidate(
        self,
        candidate_metric: float,
        champion_metric: float | None,
        promotion_delta: float,
    ) -> PromotionDecision:

        if champion_metric is None:
            metric_delta = None
            reason = "No champion model existed yet."
        else:
            metric_delta = candidate_metric - champion_metric
            if metric_delta > promotion_delta:
                reason = "Candidate model better than former champion (quality metric improvement above required threshold)."
            else:
                reason = "Candidate model does not achieve required quality metric improvement."

        return PromotionDecision(
            candidate_metric=candidate_metric,
            champion_metric=champion_metric,
            metric_delta=metric_delta,
            required_delta=promotion_delta,
            reason=reason,
        )

    def promote_candidate(
        self,
        decision: PromotionDecision,
        registry: ModelRegistry,
        model_name: str,
        candidate_version: int,
    ) -> None:
        """Demote current champion model and promote model candidate to champion."""

        if not decision.promote:
            raise ValueError(
                "A candidate can only be promoted, if it was evaluated and the evaluation was positive."
            )

        if decision.champion_metric is not None:
            # demote current champion if it existed
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
