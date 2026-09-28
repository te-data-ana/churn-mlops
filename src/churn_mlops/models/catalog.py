from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


def lin_svc(**kwargs):
    return CalibratedClassifierCV(
        estimator=LinearSVC(
            C=kwargs.get("C", 1.0),
            random_state=kwargs.get("random_state", DEFAULT_RANDOM_STATE),
        ),
        method=kwargs.get("method", "sigmoid"),
        cv=kwargs.get("cv", 5),
        ensemble=False,
    )


def rbf_svc(**kwargs):
    return CalibratedClassifierCV(
        estimator=SVC(
            kernel="rbf",
            C=kwargs.get("C", 1.0),
            random_state=kwargs.get("random_state", DEFAULT_RANDOM_STATE),
        ),
        method=kwargs.get("method", "sigmoid"),
        cv=kwargs.get("cv", 5),
        ensemble=False,
    )


# default model parameters
DEFAULT_RANDOM_STATE = 26

MODEL_CATALOG = {
    "dc": {
        "clf": DummyClassifier,
        "default_params": {"strategy": "most_frequent"},
    },
    "nb": {
        "clf": GaussianNB,
        "default_params": {},
    },
    "lr": {
        "clf": LogisticRegression,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
            "max_iter": 100,
        },
    },
    "lsvc": {
        "clf": lin_svc,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
    },
    "svc": {
        "clf": rbf_svc,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
    },
    "dt": {
        "clf": DecisionTreeClassifier,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
    },
    "ada": {
        "clf": AdaBoostClassifier,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
            "n_estimators": 100,
        },
    },
    "et": {
        "clf": ExtraTreesClassifier,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
    },
    "rf": {
        "clf": RandomForestClassifier,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
    },
    "hgb": {
        "clf": HistGradientBoostingClassifier,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
    },
    "knn": {
        "clf": KNeighborsClassifier,
        "default_params": {
            "n_neighbors": 5,
            "weights": "uniform",
        },
    },
    "lda": {
        "clf": LinearDiscriminantAnalysis,
        "default_params": {},
    },
    "qda": {
        "clf": QuadraticDiscriminantAnalysis,
        "default_params": {
            "reg_param": 0.1,
        },
    },
}
