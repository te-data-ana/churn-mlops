from collections.abc import Sequence

import pandas as pd


def select_feature_columns(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    data_name: str,
) -> pd.DataFrame:
    """Return only declared model features, failing on missing columns.

    Args:
        df: DataFrame containing model features and any passthrough columns.
        feature_columns: Ordered use-case model feature names.
        data_name: Label used to identify the failing data split.

    Returns:
        DataFrame with only the requested model features, in contract order.

    Raises:
        ValueError: If any declared model feature is missing.
    """
    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{data_name} data is missing required use-case feature columns: "
            f"{', '.join(missing_columns)}."
        )
    return df.loc[:, list(feature_columns)]
