from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from churn_mlops.tracking.promotion import PromotionDecision, PromotionService


@pytest.mark.unit
def test_weighted_promotion_allows_first_champion() -> None:
    decision = PromotionService().evaluate_weighted_candidate(
        candidate_metrics={"roc_auc": 0.8},
        champion_metrics=None,
        metric_weights={"roc_auc": 1.0},
        min_delta=0.02,
    )

    assert decision.promote
    assert decision.candidate_score == pytest.approx(0.8)
    assert decision.champion_score is None
    assert decision.score_delta is None
    assert decision.min_delta == pytest.approx(0.02)


@pytest.mark.unit
def test_weighted_promotion_requires_strict_score_improvement() -> None:
    service = PromotionService()
    common = {
        "candidate_metrics": {"roc_auc": 0.9},
        "champion_metrics": {"roc_auc": 0.8},
        "metric_weights": {"roc_auc": 1.0},
        "min_delta": 0.1,
    }

    equal_decision = service.evaluate_weighted_candidate(**common)
    assert not equal_decision.promote
    assert equal_decision.score_delta == pytest.approx(0.1)

    above_decision = service.evaluate_weighted_candidate(
        **(common | {"min_delta": 0.099})
    )
    assert above_decision.promote


@pytest.mark.unit
def test_weighted_promotion_rejects_insufficient_improvement() -> None:
    decision = PromotionService().evaluate_weighted_candidate(
        candidate_metrics={"roc_auc": 0.81},
        champion_metrics={"roc_auc": 0.8},
        metric_weights={"roc_auc": 1.0},
        min_delta=0.02,
    )

    assert not decision.promote
    assert decision.score_delta == pytest.approx(0.01)


@pytest.mark.unit
def test_weighted_promotion_rewards_calibration_and_ranking_metrics() -> None:
    weights = {
        "roc_auc": 0.2,
        "pr_auc": 0.2,
        "f1": 0.2,
        "brier_score": 0.2,
        "calibration_error": 0.2,
    }
    candidate_metrics = {
        "roc_auc": 0.95,
        "pr_auc": 0.9,
        "f1": 0.8,
        "brier_score": 0.1,
        "calibration_error": 0.05,
    }
    champion_metrics = {
        "roc_auc": 0.9,
        "pr_auc": 0.9,
        "f1": 0.8,
        "brier_score": 0.2,
        "calibration_error": 0.1,
    }

    decision = PromotionService().evaluate_weighted_candidate(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        metric_weights=weights,
        min_delta=0.01,
    )

    assert decision.promote
    assert decision.candidate_score == pytest.approx(0.9)
    assert decision.champion_score == pytest.approx(0.86)
    assert decision.score_delta == pytest.approx(0.04)


@pytest.mark.unit
def test_weighted_promotion_blocks_champion_missing_required_metrics() -> None:
    decision = PromotionService().evaluate_weighted_candidate(
        candidate_metrics={"roc_auc": 0.9, "calibration_error": 0.1},
        champion_metrics={"roc_auc": 0.88},
        metric_weights={"roc_auc": 0.5, "calibration_error": 0.5},
        min_delta=0.01,
    )

    assert not decision.promote
    assert decision.champion_score is None
    assert decision.score_delta is None
    assert "missing required metrics" in decision.reason


@pytest.mark.unit
def test_promote_candidate_first_champion() -> None:
    registry = Mock()
    decision = PromotionDecision(
        candidate_score=0.8,
        champion_score=None,
        score_delta=None,
        min_delta=0.01,
        promote=True,
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
        candidate_score=0.77,
        champion_score=0.75,
        score_delta=0.02,
        min_delta=0.01,
        promote=True,
        reason="better",
    )

    PromotionService().promote_candidate(
        decision=decision,
        registry=registry,
        model_name="my_model",
        candidate_version=4,
    )

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
        candidate_score=0.81,
        champion_score=0.8,
        score_delta=0.01,
        min_delta=0.02,
        promote=False,
        reason="rejected",
    )

    with pytest.raises(ValueError):
        PromotionService().promote_candidate(
            decision=decision,
            registry=object(),
            model_name="my_model",
            candidate_version=3,
        )
