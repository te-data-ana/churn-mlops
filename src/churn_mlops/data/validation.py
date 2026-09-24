import logging

import pandas as pd

from churn_mlops.data.schemas import DataSchema

logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a DataFrame against the data schema.

    Args:
        df: DataFrame containing features and optionally a binary ``churn``
            target and/or a ``reference_date`` column.

    Returns:
        The validated and type-coerced DataFrame.

    Raises:
        pandera.errors.SchemaError: If the DataFrame violates the schema.
    """
    try:
        logger.info("Validating data schema for %d rows.", len(df))
        validated = DataSchema.validate(df)
        logger.info("Data validation passed for %d rows.", len(validated))
        return validated
    except Exception:
        logger.exception("Data validation failed.")
        raise
