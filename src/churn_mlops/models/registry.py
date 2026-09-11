from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# default model parameters
DEFAULT_RANDOM_STATE = 26
MODEL_REGISTRY = {
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
        "default_params": {"random_state": DEFAULT_RANDOM_STATE, "max_iter": 200},
    },
    "dt": {
        "clf": DecisionTreeClassifier,
        "default_params": {"random_state": DEFAULT_RANDOM_STATE},
    },
    "et": {
        "clf": ExtraTreesClassifier,
        "default_params": {"random_state": DEFAULT_RANDOM_STATE},
    },
    "rf": {
        "clf": RandomForestClassifier,
        "default_params": {"random_state": DEFAULT_RANDOM_STATE},
    },
    "hgb": {
        "clf": HistGradientBoostingClassifier,
        "default_params": {"random_state": DEFAULT_RANDOM_STATE},
    },
}
