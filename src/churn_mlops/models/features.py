from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class IdentityFeatureBuilder(BaseEstimator, TransformerMixin):
    """Pass input columns through when a use case has no custom features."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit without learning state."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the input frame."""
        return X.copy()
