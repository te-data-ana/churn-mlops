import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churn_mlops.models import build_classifier_pipeline


def test_pipeline_can_fit(sample_training_data):
    X, y = sample_training_data

    pipeline = build_classifier_pipeline("lr")

    pipeline.fit(X, y)


def test_pipeline_can_predict(sample_training_data):
    X, y = sample_training_data

    pipeline = build_classifier_pipeline("lr")

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)


def test_override_params():
    pipeline = build_classifier_pipeline("rf", classifier_params={"n_estimators": 123})

    clf = pipeline.named_steps["classifier"]

    assert clf.n_estimators == 123


def test_invalid_classifier():
    with pytest.raises(ValueError):
        build_classifier_pipeline("invalid")


def test_pipeline_contains_expected_steps():
    pipeline = build_classifier_pipeline("lr")

    assert set(pipeline.named_steps) == {
        "features",
        "preprocessor",
        "classifier",
    }


def test_lr_classifier_type():
    pipeline = build_classifier_pipeline("lr")

    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_rf_classifier_type():
    pipeline = build_classifier_pipeline("rf")

    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)


def test_hgb_classifier_type():
    pipeline = build_classifier_pipeline("hgb")

    assert isinstance(
        pipeline.named_steps["classifier"], HistGradientBoostingClassifier
    )
