import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from churn_mlops.models import build_classifier_pipeline
from churn_mlops.models.pipelines import _requires_scaling


@pytest.mark.parametrize(
    "classifier",
    [
        LogisticRegression(),
        KNeighborsClassifier(),
        SVC(),
    ],
)
def test_requires_scaling(classifier):
    assert _requires_scaling(classifier) is True


@pytest.mark.parametrize(
    "classifier",
    [
        DummyClassifier(),
        ExtraTreesClassifier(),
        HistGradientBoostingClassifier(),
        RandomForestClassifier(),
        GaussianNB(),
        DecisionTreeClassifier(),
    ],
)
def test_does_not_require_scaling(classifier):
    assert _requires_scaling(classifier) is False


def test_pipeline_fit_and_predict(sample_training_data):
    X, y = sample_training_data

    pipeline = build_classifier_pipeline(LogisticRegression())

    with pytest.raises(NotFittedError):
        pipeline.predict(X)

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)


def test_pipeline_contains_expected_steps():
    pipeline = build_classifier_pipeline(LogisticRegression())

    assert set(pipeline.named_steps) == {
        "features",
        "preprocessor",
        "classifier",
    }


@pytest.mark.parametrize(
    "classifier",
    [
        DummyClassifier(),
        LogisticRegression(),
        ExtraTreesClassifier(),
        HistGradientBoostingClassifier(),
        RandomForestClassifier(),
        GaussianNB(),
        DecisionTreeClassifier(),
    ],
)
def test_pipeline_contains_classifier(classifier):
    pipeline = build_classifier_pipeline(classifier)

    assert pipeline.named_steps["classifier"] is classifier
