import pytest
from sklearn.svm import SVC, LinearSVC

from churn_mlops.models import create_model


@pytest.mark.unit
@pytest.mark.parametrize(
    ("alias", "expected_type"),
    [("lsvc", LinearSVC), ("svc", SVC)],
)
def test_svm_catalog_returns_unwrapped_estimators(
    alias: str,
    expected_type: type[LinearSVC] | type[SVC],
) -> None:
    classifier, config = create_model(alias, {})

    assert type(classifier) is expected_type
    assert config["model_name"] == expected_type.__name__
    if alias == "svc":
        assert classifier.kernel == "rbf"
