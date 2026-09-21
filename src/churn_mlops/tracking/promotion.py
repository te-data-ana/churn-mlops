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
        """Whether the candidate satisfies the promotion requirement.

        Returns:
            ``True`` when there is no champion or the metric improvement is
            strictly greater than ``required_delta``; otherwise ``False``.
        """
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
        """Evaluate a candidate metric against the current champion.

        Args:
            candidate_metric: Quality metric produced by the candidate model.
            champion_metric: Quality metric of the current champion, or
                ``None`` when no champion exists.
            promotion_delta: Minimum strict improvement required for promotion.

        Returns:
            Promotion decision containing the metric delta and explanation.
        """

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

        if decision.champion_metric is not None:
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
