from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# direcory names
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# definition of data types
DTYPES = {
    "CustomerID": "Int64",
    "Age": "Int64",
    "Gender": "category",
    "Tenure": "Int64",
    "Usage Frequency": "Int64",
    "Support Calls": "Int64",
    "Payment Delay": "Int64",
    "Subscription Type": "category",
    "Contract Length": "category",
    "Total Spend": "Float64",
    "Last Interaction": "Int64",
    "Churn": "Int64",
}

# preprocessing parameters
NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"

# default model parameters
DEFAULT_RANDOM_STATE = 26
BASELINE_CLASSIFIERS = {
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
