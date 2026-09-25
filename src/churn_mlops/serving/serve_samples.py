import argparse
import random
from pathlib import Path

import pandas as pd
import requests

from churn_mlops.config import RuntimeSettings, configure_logging
from churn_mlops.data.ingestion import load_raw_data
from churn_mlops.data.validation import validate_data


def load_sample_data(
    sample_size: int,
    random_state: int,
    settings: RuntimeSettings,
) -> pd.DataFrame:
    """Load, validate, and sample inference data.

    Reads the inference dataset, validates its contents against the
    inference schema, and returns a random sample of records.

    Args:
        sample_size: Number of records to sample.
        random_state: Seed used for reproducible sampling.
        settings: Runtime configuration containing data locations.

    Returns:
        A validated DataFrame containing the sampled inference records.
    """

    df = load_raw_data(
        file_name="inference.csv",
        index_col="customerid",
        data_dir=settings.raw_data_dir,
    )

    df = validate_data(df)

    return df.sample(
        n=sample_size,
        random_state=random_state,
    )


def predict_samples(
    df: pd.DataFrame,
    settings: RuntimeSettings,
) -> pd.DataFrame:
    """Generate predictions for a collection of samples.

    Sends each row of the provided DataFrame to the prediction API and
    combines the returned prediction results with the original input
    features.

    Args:
        df: DataFrame containing inference records.
        settings: Runtime configuration containing API connection
            details.

    Returns:
        A DataFrame containing the original input features together with
        prediction results returned by the API.

    Raises:
        requests.HTTPError: If any prediction request returns a non-
            successful status code.
    """

    predictions = []

    for _, row in df.iterrows():
        response = requests.post(
            f"http://{settings.api_host}:{settings.api_port}/predict",
            json=row.to_dict(),
        )

        response.raise_for_status()

        predictions.append(response.json())

    return pd.concat(
        [
            df,
            pd.DataFrame(predictions, index=df.index),
        ],
        axis=1,
    )


def save_predictions(
    df: pd.DataFrame,
    run_id: str,
    settings: RuntimeSettings,
) -> Path:
    """Persist prediction results to a CSV file.

    Writes the provided DataFrame to the configured output directory
    using the naming convention
    ``{run_id}_sample_predictions.csv``.

    Args:
        df: DataFrame containing prediction results.
        run_id: Identifier used as the filename prefix.
        settings: Runtime configuration containing output locations.

    Returns:
        Path to the generated CSV file.
    """

    output_path = settings.output_dir / f"{run_id}_sample_predictions.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.reset_index(drop=True).to_csv(
        output_path,
        index=False,
    )

    return output_path


def serve_samples(
    sample_size: int,
    random_state: int,
    settings: RuntimeSettings | None = None,
) -> Path:
    """Generate and store predictions for sampled inference data.

    Loads and validates inference data, draws a random sample of records,
    submits the records to the prediction API, and stores the resulting
    predictions in a CSV file.

    Args:
        sample_size: Number of records to sample and score.
        random_state: Seed used for reproducible sampling and run
            identification.
        settings: Runtime configuration. If not provided, a default
            configuration is created.

    Returns:
        Path to the generated prediction CSV file.

    Raises:
        requests.HTTPError: If any prediction request returns a non-
            successful status code.
    """

    settings = settings or RuntimeSettings()

    df_sample = load_sample_data(
        sample_size=sample_size,
        random_state=random_state,
        settings=settings,
    )

    df_predictions = predict_samples(
        df=df_sample,
        settings=settings,
    )

    run_id = f"{random_state:04d}"

    return save_predictions(
        df=df_predictions,
        run_id=run_id,
        settings=settings,
    )


def main() -> None:
    """Generate predictions for a random sample of inference data.

    Reads inference data, validates it, samples a user-specified number
    of records, and sends each record to the prediction API. The
    resulting predictions are combined with the sampled input data and
    written to a CSV file named with a four-digit run identifier. If no
    random state is provided, a random seed is generated.

    Raises:
        requests.HTTPError: If any prediction request returns a non-
            successful status code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", required=True)
    parser.add_argument("--random_state")

    args = parser.parse_args()

    random_state = int(args.random_state or random.randint(1, 1000))

    serve_samples(
        sample_size=int(args.sample_size),
        random_state=random_state,
    )


if __name__ == "__main__":
    configure_logging()
    main()
