from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from churn_mlops.tracking.promotion import PromotionDecision, PromotionService


@pytest.mark.unit
def test_promotion_decision_positive_without_champion() -> None:
    decision = PromotionDecision(
        candidate_metric=0.82,
        champion_metric=None,
        metric_delta=None,
        required_delta=0.01,
        reason="No champion.",
    )

    assert decision.promote is True


@pytest.mark.unit
def test_promotion_decision_positive_if_required_delta_exeeded() -> None:
    decision = PromotionDecision(
        candidate_metric=0.86,
        champion_metric=0.84,
        metric_delta=0.02,
        required_delta=0.01,
        reason="better",
    )

    assert decision.promote is True


@pytest.mark.unit
def test_promotion_decision_rejection_below_required_delta() -> None:
    decision = PromotionDecision(
        candidate_metric=0.845,
        champion_metric=0.84,
        metric_delta=0.005,
        required_delta=0.02,
        reason="not enough",
    )

    assert decision.promote is False


@pytest.mark.unit
def test_evaluate_candidate_promotes_without_champion() -> None:
    decision = PromotionService().evaluate_candidate(
        candidate_metric=0.80,
        champion_metric=None,
        promotion_delta=0.02,
    )

    assert decision.promote is True
    assert decision.metric_delta is None


@pytest.mark.unit
def test_evaluate_candidate_promotes_sufficient_improvement() -> None:
    decision = PromotionService().evaluate_candidate(
        candidate_metric=0.90,
        champion_metric=0.80,
        promotion_delta=0.02,
    )

    assert decision.promote is True
    assert decision.metric_delta == decision.candidate_metric - decision.champion_metric


@pytest.mark.unit
def test_evaluate_candidate_rejects_insufficient_improvement() -> None:
    decision = PromotionService().evaluate_candidate(
        candidate_metric=0.81,
        champion_metric=0.80,
        promotion_delta=0.02,
    )

    assert decision.promote is False


@pytest.mark.unit
def test_promote_candidate_first_champion() -> None:
    registry = Mock()
    decision = PromotionDecision(
        candidate_metric=0.8,
        champion_metric=None,
        metric_delta=None,
        required_delta=0.01,
        reason="first champion",
    )

    PromotionService().promote_candidate(
        decision=decision,
        registry=registry,
        model_name="my_model",
        candidate_version=1,
    )

    registry.set_alias.assert_called_once_with(
        model_name="my_model",
        alias="champion",
        version=1,
    )


@pytest.mark.unit
def test_promote_candidate_replaces_champion() -> None:
    registry = Mock()
    registry.get_champion_version.return_value = SimpleNamespace(version=2)
    decision = PromotionDecision(
        candidate_metric=0.77,
        champion_metric=0.75,
        metric_delta=0.02,
        required_delta=0.01,
        reason="better",
    )

    # Retire the current champion before assigning the candidate as champion.
    PromotionService().promote_candidate(
        decision=decision,
        registry=registry,
        model_name="my_model",
        candidate_version=4,
    )

    assert registry.set_alias.call_count == 2
    assert registry.set_alias.call_args_list == [
        call(
            model_name="my_model",
            alias="former_champion",
            version=2,
        ),
        call(
            model_name="my_model",
            alias="champion",
            version=4,
        ),
    ]


@pytest.mark.unit
def test_promote_candidate_rejection() -> None:
    decision = PromotionDecision(
        candidate_metric=0.81,
        champion_metric=0.80,
        metric_delta=0.01,
        required_delta=0.02,
        reason="rejected",
    )

    with pytest.raises(ValueError):
        PromotionService().promote_candidate(
            decision=decision,
            registry=object(),
            model_name="my_model",
            candidate_version=3,
        )
