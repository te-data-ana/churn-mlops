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
        "search_space": {
            "var_smoothing": {
                "type": "float",
                "low": 1e-12,
                "high": 1e-6,
                "log": True,
            },
        },
    },
    "lr": {
        "clf": LogisticRegression,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
            "max_iter": 100,
        },
        "search_space": {
            "C": {"type": "float", "low": 1e-4, "high": 1e2, "log": True},
        },
    },
    "lsvc": {
        "clf": lin_svc,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
        "search_space": {
            "C": {"type": "float", "low": 1e-4, "high": 1e2, "log": True},
        },
    },
    "svc": {
        "clf": rbf_svc,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
        "search_space": {
            "C": {"type": "float", "low": 1e-4, "high": 1e2, "log": True},
        },
    },
    "dt": {
        "clf": DecisionTreeClassifier,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
        "search_space": {
            "max_depth": {
                "type": "categorical",
                "choices": [None, 3, 5, 10, 20],
            },
            "min_samples_leaf": {"type": "int", "low": 5, "high": 100},
        },
    },
    "ada": {
        "clf": AdaBoostClassifier,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
            "n_estimators": 100,
        },
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 2.0,
                "log": True,
            },
        },
    },
    "et": {
        "clf": ExtraTreesClassifier,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {
                "type": "categorical",
                "choices": [None, 5, 10, 20],
            },
            "min_samples_leaf": {"type": "int", "low": 5, "high": 100},
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", "log2", None],
            },
        },
    },
    "rf": {
        "clf": RandomForestClassifier,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
        "search_space": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {
                "type": "categorical",
                "choices": [None, 5, 10, 20],
            },
            "min_samples_leaf": {"type": "int", "low": 5, "high": 100},
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", "log2", None],
            },
        },
    },
    "hgb": {
        "clf": HistGradientBoostingClassifier,
        "default_params": {
            "random_state": DEFAULT_RANDOM_STATE,
        },
        "search_space": {
            "max_iter": {"type": "int", "low": 50, "high": 500},
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.3,
                "log": True,
            },
            "max_leaf_nodes": {"type": "int", "low": 16, "high": 64},
            "l2_regularization": {
                "type": "float",
                "low": 1e-8,
                "high": 10.0,
                "log": True,
            },
        },
    },
    "knn": {
        "clf": KNeighborsClassifier,
        "default_params": {
            "n_neighbors": 5,
            "weights": "uniform",
        },
        "search_space": {
            "n_neighbors": {"type": "int", "low": 3, "high": 25},
            "weights": {
                "type": "categorical",
                "choices": ["uniform", "distance"],
            },
        },
    },
    "lda": {
        "clf": LinearDiscriminantAnalysis,
        "default_params": {},
        "search_space": {
            "tol": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
        },
    },
    "qda": {
        "clf": QuadraticDiscriminantAnalysis,
        "default_params": {
            "reg_param": 0.1,
        },
        "search_space": {
            "reg_param": {"type": "float", "low": 0.0, "high": 1.0},
        },
    },
}
