from collections.abc import Callable
from typing import Literal
from unittest.mock import Mock

import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from churn_mlops.config.schemas import TrainingConfig
from churn_mlops.evaluation import select_f1_threshold
from churn_mlops.training import TrainingResult, model_training, train_model


def assert_training_result(result) -> None:
    assert isinstance(result, TrainingResult)
    assert isinstance(result.trained_pipeline, Pipeline)

    assert result.metrics
    assert result.feature_names_in
    assert result.feature_names_out

    metrics = result.metrics
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["brier_score"] <= 1
    assert metrics["fit_time_sec"] >= 0


@pytest.mark.integration
def test_run_training_job_returns_valid_training_result(
    registered_model: dict[str, object],
) -> None:
    assert_training_result(registered_model["result"])


@pytest.mark.unit
@pytest.mark.parametrize("classifier", ["lr", "dt", "rf", "hgb"])
def test_train_returns_valid_result_for_supported_classifier(
    classifier: Literal["lr", "dt", "rf", "hgb"],
    config_factory: Callable[..., TrainingConfig],
    temporal_sample_training_df: pd.DataFrame,
):
    cfg = config_factory(classifier=classifier)
    train_df = temporal_sample_training_df.iloc[:6]
    test_df = temporal_sample_training_df.iloc[6:]
    result = train_model(cfg, train_df, test_df)

    assert_training_result(result)

    assert len(result.feature_names_out) > len(result.feature_names_in)


@pytest.mark.unit
@pytest.mark.parametrize("classifier", ["lr", "dt", "rf", "hgb"])
def test_trained_pipeline_predicts_classes_and_probabilities(
    classifier: Literal["lr", "dt", "rf", "hgb"],
    config_factory: Callable[..., TrainingConfig],
    temporal_sample_training_df: pd.DataFrame,
) -> None:
    cfg = config_factory(classifier=classifier)
    train_df = temporal_sample_training_df.iloc[:6]
    test_df = temporal_sample_training_df.iloc[6:]
    result = train_model(cfg, train_df, test_df)

    X = temporal_sample_training_df.iloc[:6].drop(columns=["churn"])

    predictions = result.trained_pipeline.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})

    probas = result.trained_pipeline.predict_proba(X)
    assert probas.shape[0] == len(X)
    assert probas.shape[1] == 2
    assert ((probas >= 0) & (probas <= 1)).all()


@pytest.mark.unit
def test_train_rejects_unknown_classifier(
    config_factory: Callable[..., TrainingConfig],
    temporal_sample_training_df: pd.DataFrame,
) -> None:

    cfg = config_factory(classifier="invalid")

    with pytest.raises(ValueError, match="Unknown classifier"):
        train_df = temporal_sample_training_df.iloc[:6]
        test_df = temporal_sample_training_df.iloc[6:]
        train_model(cfg, train_df, test_df)


@pytest.mark.unit
def test_train_returns_reproducible_metrics_for_same_input(
    config_factory: Callable[..., TrainingConfig],
    temporal_sample_training_df: pd.DataFrame,
) -> None:
    cfg = config_factory(classifier="dt")

    # Train twice with identical inputs to verify deterministic results.
    train_df = temporal_sample_training_df.iloc[:6]
    test_df = temporal_sample_training_df.iloc[6:]
    result1 = train_model(cfg, train_df, test_df)
    result2 = train_model(cfg, train_df, test_df)

    assert result1.metrics["roc_auc"] == result2.metrics["roc_auc"]
    assert result1.metrics["accuracy"] == result2.metrics["accuracy"]


@pytest.mark.unit
def test_train_uses_override_and_excludes_columns(
    config_factory: Callable[..., TrainingConfig],
    sample_training_df: pd.DataFrame,
) -> None:
    cfg = config_factory(classifier="dt")
    dated_df = sample_training_df.copy()
    dated_df["reference_date"] = pd.date_range("2026-01-01", periods=len(dated_df))
    dated_df["customerid"] = range(len(dated_df))

    result = train_model(
        config=cfg,
        train_df=dated_df.iloc[:7],
        test_df=dated_df.iloc[7:],
        model_params_override={"max_depth": 2},
        exclude_columns=["reference_date", "customerid"],
    )

    assert result.classifier_config["max_depth"] == 2
    assert "reference_date" not in result.feature_names_in
    assert "customerid" not in result.feature_names_in


@pytest.mark.unit
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_train_fits_time_aware_calibrated_classifier(
    method: str,
    config_factory: Callable[..., TrainingConfig],
    sample_training_df: pd.DataFrame,
) -> None:
    config = config_factory(classifier="lr")
    config.calibration.method = method
    train_df = pd.concat([sample_training_df] * 8, ignore_index=True)
    train_df["reference_date"] = pd.date_range(
        "2026-01-01",
        periods=len(train_df),
    )

    result = train_model(
        config,
        train_df,
        sample_training_df.copy(),
    )

    assert isinstance(result.trained_pipeline, CalibratedClassifierCV)
    assert config.calibration.method == method
    assert 0 <= result.threshold <= 1
    assert 0 <= result.metrics["calibration_error"] <= 1


@pytest.mark.unit
def test_train_fits_once_and_selects_threshold_from_oot_test(
    monkeypatch: pytest.MonkeyPatch,
    config_factory: Callable[..., TrainingConfig],
    temporal_sample_training_df: pd.DataFrame,
) -> None:
    config = config_factory(classifier="dt")
    train_df = temporal_sample_training_df.iloc[:6]
    test_df = temporal_sample_training_df.iloc[6:]
    fit_spy = Mock(wraps=model_training.fit_estimator_with_calibration)
    monkeypatch.setattr(model_training, "fit_estimator_with_calibration", fit_spy)

    result = train_model(config, train_df, test_df)

    X_test = test_df.drop(columns=["churn", "reference_date"])
    probabilities = result.trained_pipeline.predict_proba(X_test)[:, 1]
    expected_threshold = select_f1_threshold(test_df["churn"], probabilities)

    fit_spy.assert_called_once()
    assert result.threshold == pytest.approx(expected_threshold)
    assert result.threshold_data == "out_of_time_test"
