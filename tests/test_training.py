from datetime import datetime

import pytest
from sklearn.pipeline import Pipeline

from churn_mlops.training import TrainingResult, train, train_from_files


def assert_training_result(result):
    assert isinstance(result, TrainingResult)
    assert isinstance(result.trained_pipeline, Pipeline)

    assert result.metrics
    assert result.metadata

    metrics = result.metrics
    assert 0.5 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["brier_score"] <= 1
    assert metrics["fit_time_sec"] >= 0

    metadata = result.metadata
    assert datetime.fromisoformat(metadata["timestamp"])
    assert metadata["training_rows"] > 0
    assert len(metadata["feature_names_in"]) > 0
    assert len(metadata["feature_names_out"]) > 0


@pytest.mark.integration
def test_training_integration():
    result = train_from_files()

    assert_training_result(result)


@pytest.mark.parametrize("classifier", ["lr", "dt", "rf", "hgb"])
def test_training_returns_results(classifier, config_factory, sample_training_df):
    cfg = config_factory(classifier=classifier)

    result = train(cfg, sample_training_df)

    assert_training_result(result)

    assert len(result.metadata["feature_names_out"]) > len(
        result.metadata["feature_names_in"]
    )


@pytest.mark.parametrize("classifier", ["lr", "dt", "rf", "hgb"])
def test_training_pipeline_can_predict(classifier, config_factory, sample_training_df):
    cfg = config_factory(classifier=classifier)
    result = train(cfg, sample_training_df)

    X = sample_training_df.drop(columns=["churn"])

    predictions = result.trained_pipeline.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})

    probas = result.trained_pipeline.predict_proba(X)
    assert probas.shape[0] == len(X)
    assert probas.shape[1] == 2
    assert ((probas >= 0) & (probas <= 1)).all()


def test_unknown_classifier_error(config_factory, sample_training_df):

    cfg = config_factory(classifier="invalid")

    with pytest.raises(ValueError, match="Unknown classifier"):
        train(cfg, sample_training_df)


def test_training_reproducible(config_factory, sample_training_df):
    cfg = config_factory(classifier="dt")

    result1 = train(cfg, sample_training_df)
    result2 = train(cfg, sample_training_df)

    assert result1.metrics["roc_auc"] == result2.metrics["roc_auc"]
    assert result1.metrics["accuracy"] == result2.metrics["accuracy"]
    assert result1.metadata["training_rows"] == result2.metadata["training_rows"]
