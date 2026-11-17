from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class UseCase:
    """Data and adapter contract for one binary-classification use case."""

    name: str
    target_column: str
    time_column: str
    feature_columns: tuple[str, ...]
    data_validator: Callable[[pd.DataFrame], pd.DataFrame]
    feature_builder_factory: Callable[[Mapping[str, Any] | None], BaseEstimator]
    request_schema: type[BaseModel] | None = None
    index_column: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Use-case name must not be empty.")
        if not self.target_column or not self.time_column:
            raise ValueError("Use-case target and time columns must not be empty.")
        if not self.feature_columns:
            raise ValueError("Use-case feature_columns must not be empty.")
        if {self.target_column, self.time_column}.intersection(self.feature_columns):
            raise ValueError(
                "Use-case feature_columns must not include target or time columns."
            )

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize a dataframe using this use case's data contract."""
        return self.data_validator(df)

    def build_feature_builder(self) -> BaseEstimator:
        """Create a fresh scikit-learn-compatible feature transformer."""
        return self.feature_builder_factory()
