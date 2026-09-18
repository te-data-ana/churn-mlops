from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
from mlflow.exceptions import MlflowException

import churn_mlops.tracking.mlflow as tracking
from churn_mlops.config import TMP_DIR
from churn_mlops.tracking.promotion import PromotionDecision, PromotionService


@pytest.mark.unit
def test_setup_local_experiment_creates_new(monkeypatch):
    monkeypatch.setattr(
        tracking.mlflow, "create_experiment", lambda *args, **kwargs: "123"
    )

    mock_set_experiment = MagicMock()

    monkeypatch.setattr(tracking.mlflow, "set_experiment", mock_set_experiment)

    experiment_id = tracking.setup_local_experiment("test")

    assert experiment_id == "123"
    mock_set_experiment.assert_called_once()


@pytest.mark.unit
def test_setup_local_experiment_existing(monkeypatch):

    def raise_exc(*args, **kwargs):
        raise MlflowException("experiment already exists")

    monkeypatch.setattr(tracking.mlflow, "create_experiment", raise_exc)

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "123"

    monkeypatch.setattr(
        tracking.mlflow, "get_experiment_by_name", lambda _: mock_experiment
    )

    mock_set_experiment = MagicMock()

    monkeypatch.setattr(tracking.mlflow, "set_experiment", mock_set_experiment)

    experiment_id = tracking.setup_local_experiment("test")

    assert experiment_id == "123"


@pytest.mark.unit
def test_log_experiment_result(
    monkeypatch,
    mock_taining_result,
    config_factory,
):
    config_file_path = TMP_DIR / "test.yaml"
    config_file_path.write_text("test")

    mock_log_metrics = MagicMock()
    mock_log_param = MagicMock()
    mock_log_params = MagicMock()
    mock_set_tags = MagicMock()
    mock_log_model = MagicMock()
    mock_log_artifact = MagicMock()

    monkeypatch.setattr(tracking.mlflow, "log_metrics", mock_log_metrics)
    monkeypatch.setattr(tracking.mlflow, "log_param", mock_log_param)
    monkeypatch.setattr(tracking.mlflow, "log_params", mock_log_params)
    monkeypatch.setattr(tracking.mlflow, "set_tags", mock_set_tags)
    monkeypatch.setattr(tracking.mlflow.sklearn, "log_model", mock_log_model)
    monkeypatch.setattr(tracking.mlflow, "log_artifact", mock_log_artifact)

    tracking.log_experiment_result(
        mock_taining_result, config_factory(), config_file_path
    )

    mock_log_metrics.assert_called_once()
    mock_log_param.assert_called_once()
    assert mock_log_params.call_count > 4
    mock_set_tags.assert_called_once()
    mock_log_model.assert_called_once()
    assert mock_log_artifact.call_count >= 2


@pytest.mark.unit
def test_registry_set_alias_calls_mlflow_client(registry):
    registry.set_alias(
        model_name="my_model",
        alias="champion",
        version=5,
    )

    registry.client.set_registered_model_alias.assert_called_once_with(
        name="my_model",
        alias="champion",
        version=5,
    )


@pytest.mark.unit
def test_registry_get_model_version(registry):
    registry.get_model_version(
        model_name="my_model",
        version=3,
    )

    registry.client.get_model_version.assert_called_once_with(
        name="my_model",
        version="3",
    )


@pytest.mark.unit
def test_registry_get_metric_by_alias(registry):
    version = Mock()
    version.run_id = "abc"

    run = Mock()
    run.data.metrics = {"roc_auc": 0.84}

    registry.get_model_version_by_alias = Mock(return_value=version)

    registry.client.get_run.return_value = run

    metric = registry.get_metric_by_alias(
        model_name="my_model",
        alias="champion",
        metric_name="roc_auc",
    )

    assert metric == 0.84


@pytest.mark.unit
def test_registry_get_threshold_by_alias(registry):
    version = Mock()
    version.run_id = "abc"

    run = Mock()
    run.data.params = {"threshold": 0.4}

    registry.get_model_version_by_alias = Mock(return_value=version)

    registry.client.get_run.return_value = run

    threshold = registry.get_threshold_by_alias(
        model_name="my_model",
        alias="champion",
    )

    assert threshold == 0.4


@pytest.mark.unit
def test_registry_get_champion_version_none(registry):
    registry.get_model_version_by_alias = Mock(
        side_effect=MlflowException("no model registered with alias='champion'")
    )

    champion = registry.get_champion_version("my_model")

    assert champion is None


@pytest.mark.unit
def test_registry_get_champion_version_success(registry):
    champion = Mock()

    registry.get_model_version_by_alias = Mock(return_value=champion)

    result = registry.get_champion_version("my_model")

    assert result is champion


@pytest.mark.unit
def test_promotion_if_no_champion():
    decision = PromotionDecision(
        candidate_metric=0.82,
        champion_metric=None,
        metric_delta=None,
        required_delta=0.01,
        reason="No champion.",
    )

    assert decision.promote is True


@pytest.mark.unit
def test_promotion_if_delta_exceeds_threshold():
    decision = PromotionDecision(
        candidate_metric=0.86,
        champion_metric=0.84,
        metric_delta=0.02,
        required_delta=0.01,
        reason="better",
    )

    assert decision.promote is True


@pytest.mark.unit
def test_promotion_rejects_small_improvement():
    decision = PromotionDecision(
        candidate_metric=0.845,
        champion_metric=0.84,
        metric_delta=0.005,
        required_delta=0.02,
        reason="not enough",
    )

    assert decision.promote is False


@pytest.mark.unit
def test_evaluate_candidate_without_champion():
    service = PromotionService()

    decision = service.evaluate_candidate(
        candidate_metric=0.80,
        champion_metric=None,
        promotion_delta=0.02,
    )

    assert decision.promote is True
    assert decision.metric_delta is None


@pytest.mark.unit
def test_evaluate_candidate_positive():
    service = PromotionService()

    decision = service.evaluate_candidate(
        candidate_metric=0.90,
        champion_metric=0.80,
        promotion_delta=0.02,
    )

    assert decision.promote is True
    assert decision.metric_delta == decision.candidate_metric - decision.champion_metric


@pytest.mark.unit
def test_evaluate_candidate_negative():
    service = PromotionService()

    decision = service.evaluate_candidate(
        candidate_metric=0.81,
        champion_metric=0.80,
        promotion_delta=0.02,
    )

    assert decision.promote is False


@pytest.mark.unit
def test_promote_candidate_first_champion():
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
def test_promote_candidate_replaces_champion():
    registry = Mock()

    registry.get_champion_version.return_value = SimpleNamespace(version=2)

    decision = PromotionDecision(
        candidate_metric=0.77,
        champion_metric=0.75,
        metric_delta=0.02,
        required_delta=0.01,
        reason="better",
    )

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
def test_promote_candidate_rejection():
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
