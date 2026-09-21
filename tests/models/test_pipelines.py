from typing import Any

import pytest
from pandas import Series
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "classifier",
    [
        LogisticRegression(),
        KNeighborsClassifier(),
        SVC(),
    ],
)
def test_requires_scaling_returns_true_for_scaling_classifiers(
    classifier: LogisticRegression | KNeighborsClassifier | SVC,
) -> None:
    assert _requires_scaling(classifier) is True


@pytest.mark.unit
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
def test_requires_scaling_returns_false_for_non_scaling_classifiers(
    classifier: DummyClassifier
    | ExtraTreesClassifier
    | HistGradientBoostingClassifier
    | RandomForestClassifier
    | GaussianNB
    | DecisionTreeClassifier,
) -> None:
    assert _requires_scaling(classifier) is False


@pytest.mark.unit
def test_classifier_pipeline_predicts_after_fitting(
    sample_training_data: tuple[Any, Series],
) -> None:
    X, y = sample_training_data

    pipeline = build_classifier_pipeline(LogisticRegression())

    # Verify the pipeline rejects prediction before it has been fitted.
    with pytest.raises(NotFittedError):
        pipeline.predict(X)

    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == len(X)


@pytest.mark.unit
def test_classifier_pipeline_contains_expected_steps() -> None:
    pipeline = build_classifier_pipeline(LogisticRegression())

    assert set(pipeline.named_steps) == {
        "features",
        "preprocessor",
        "classifier",
    }


@pytest.mark.unit
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
def test_classifier_pipeline_preserves_supplied_classifier(
    classifier: DummyClassifier
    | LogisticRegression
    | ExtraTreesClassifier
    | HistGradientBoostingClassifier
    | RandomForestClassifier
    | GaussianNB
    | DecisionTreeClassifier,
) -> None:
    pipeline = build_classifier_pipeline(classifier)

    assert pipeline.named_steps["classifier"] is classifier
