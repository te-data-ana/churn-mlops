"""Serve sample records through the prediction API."""

from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from churn_mlops.config import ServingSettings, configure_logging
from churn_mlops.data.ingestion import load_raw_data
from churn_mlops.data.validation import validate_data

API_STARTUP_TIMEOUT_SECONDS = 30
API_HEALTH_CHECK_INTERVAL_SECONDS = 1


def load_sample_data(
    sample_size: int,
    random_state: int,
    settings: ServingSettings,
    drop_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load, validate, and sample inference data.

    Reads the inference dataset, validates its contents against the
    inference schema, and returns a random sample of records.

    Args:
        sample_size: Number of records to sample.
        random_state: Seed used for reproducible sampling.
        settings: Runtime configuration containing data locations.
        drop_columns: Optional list of column names to drop from loaded data.

    Returns:
        A validated DataFrame containing the sampled inference records.
    """

    df = load_raw_data(
        file_name="inference.csv",
        index_col="customerid",
        data_dir=settings.data_dir / "raw",
        drop_columns=drop_columns,
    )

    df = validate_data(df)

    return df.sample(
        n=sample_size,
        random_state=random_state,
    )


def predict_samples(
    df: pd.DataFrame,
    settings: ServingSettings,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """Generate predictions for a collection of samples.

    Sends each row of the provided DataFrame to the prediction API and
    combines the returned prediction results with the original input
    features.

    Args:
        df: DataFrame containing inference records.
        settings: Runtime configuration containing API connection
            details.
        reference_date: Optional reference date to use for all
            prediction requests. If not provided, a 'reference_date'
            column in `df` is used when available.

    Returns:
        A DataFrame containing the original input features together with
        prediction results returned by the API.

    Raises:
        requests.HTTPError: If any prediction request returns a non-
            successful status code.
    """
    logger = logging.getLogger(__name__)

    predictions = []

    url = f"http://{settings.api_host}:{settings.api_port}/predict"
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        if index % 100 == 0:
            logger.info("Processed %s/%s samples", index, len(df))
        features = row.to_dict()
        row_reference_date = features.pop("reference_date", None)

        effective_reference_date = (
            reference_date if reference_date is not None else row_reference_date
        )

        params = (
            {"reference_date": pd.Timestamp(effective_reference_date).isoformat()}
            if pd.notna(effective_reference_date)
            else {}
        )

        response = requests.post(url=url, json=features, params=params)
        response.raise_for_status()
        predictions.append(response.json())

    return pd.concat(
        [
            df.drop(columns=["reference_date"], errors="ignore"),
            pd.DataFrame(predictions, index=df.index),
        ],
        axis=1,
    )


def save_predictions(
    df: pd.DataFrame,
    run_id: str,
    settings: ServingSettings,
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
    settings: ServingSettings | None = None,
    reference_date: datetime | None = None,
    drop_columns: list[str] | None = None,
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
        reference_date: Optional reference date for the sampled data.
        drop_columns: Optional list of column names to drop from loaded
            data.

    Returns:
        Path to the generated prediction CSV file.

    Raises:
        requests.HTTPError: If any prediction request returns a non-
            successful status code.
    """

    settings = settings or ServingSettings()

    df_sample = load_sample_data(
        sample_size=sample_size,
        random_state=random_state,
        settings=settings,
        drop_columns=drop_columns,
    )

    df_predictions = predict_samples(
        df=df_sample,
        settings=settings,
        reference_date=reference_date,
    )

    run_id = f"{random_state:04d}"

    return save_predictions(
        df=df_predictions,
        run_id=run_id,
        settings=settings,
    )


def start_api_server(host: str, port: int) -> subprocess.Popen:
    """Start the FastAPI server in a subprocess."""
    return subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "churn_mlops.serving.api:app",
            "--host",
            host,
            "--port",
            str(port),
            "--no-access-log",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_api(
    host: str,
    port: int,
    timeout_seconds: int = API_STARTUP_TIMEOUT_SECONDS,
) -> None:
    """Wait until the API responds successfully."""

    health_url = f"http://{host}:{port}/health"

    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            response = requests.get(
                health_url,
                timeout=2,
            )

            if response.status_code == 200:
                return

        except requests.RequestException:
            pass

        time.sleep(API_HEALTH_CHECK_INTERVAL_SECONDS)

    raise TimeoutError(f"API did not become ready within {timeout_seconds} seconds.")


def stop_api_server(process: subprocess.Popen) -> None:
    """Stop the API subprocess."""

    process.terminate()

    try:
        process.wait(timeout=10)

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def custom_parser(settings: ServingSettings) -> argparse.ArgumentParser:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_size",
        default=1000,
        type=int,
        help="Number of sample to be drawn randomly and served to API.",
    )
    parser.add_argument(
        "--random_state",
        required=False,
        type=int,
        help="Random seed to allow reproducible sample draws.",
    )
    parser.add_argument(
        "--reference_date",
        type=datetime.fromisoformat,
        help="ISO-8601 reference date, e.g. '2026-09-30T12:00:00'",
    )
    parser.add_argument(
        "--drop_columns",
        nargs="*",
        default=[],
        help="Columns to remove before sending records to the API",
    )
    parser.add_argument(
        "--host",
        default=settings.api_host,
        help="Hostname for the API server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help="Port for the API server.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Registered model name (defaults to the configured serving model).",
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Exact registered model version; mutually exclusive with a manifest.",
    )
    parser.add_argument(
        "--model_manifest",
        type=Path,
        default=None,
        help="Path of model manifest in configured data directory.",
    )

    return parser


def configure_model_environment(
    *,
    args: argparse.Namespace,
    settings: ServingSettings | None = None,
) -> None:
    """Configure model-related environment variables."""

    if args.model_manifest is not None:
        from churn_mlops.tracking.manifest import TrainingManifest

        model_manifest_file = settings.data_dir / args.model_manifest
        manifest = TrainingManifest.read(model_manifest_file)

        os.environ["MODEL_NAME"] = manifest.model_name
        os.environ["MODEL_VERSION"] = str(manifest.model_version)

    if args.model_name is not None and args.model_version is not None:
        os.environ["MODEL_NAME"] = args.model_name
        os.environ["MODEL_VERSION"] = str(args.model_version)


def main() -> None:
    """Generate predictions for a random sample of inference data.

    Reads inference data, validates it, samples a user-specified number
    of records, optionally removes selected columns, and sends each
    record to the prediction API. An optional reference date can be
    provided for all prediction requests. The resulting predictions are
    combined with the sampled input data and written to a CSV file named
    with a four-digit run identifier. If no random state is provided, a
    random seed is generated.

    Raises:
        requests.HTTPError: If any prediction request returns a non-
            successful status code.
    """

    configure_logging()

    settings = ServingSettings()

    parser = custom_parser(settings=settings)
    args = parser.parse_args()

    if args.model_manifest is not None and (
        args.model_name is not None or args.model_version is not None
    ):
        parser.error(
            "--model_manifest cannot be combined with --model_name or --model_version."
        )
    if (args.model_name is None) != (args.model_version is None):
        parser.error("--model_name and --model_version must be provided together.")

    # if args.model_manifest is not None:
    #     from churn_mlops.tracking.manifest import TrainingManifest

    #     model_manifest_file = settings.data_dir / args.model_manifest
    #     manifest = TrainingManifest.read(model_manifest_file)
    #     os.environ["MODEL_NAME"] = manifest.model_name
    #     os.environ["MODEL_VERSION"] = str(manifest.model_version)

    # if args.model_name is not None and args.model_version is not None:
    #     os.environ["MODEL_NAME"] = args.model_name
    #     os.environ["MODEL_VERSION"] = str(args.model_version)

    configure_model_environment(args=args, settings=settings)

    random_state = int(args.random_state or random.randint(1, 1000))

    api_process = start_api_server(
        host=args.host,
        port=args.port,
    )

    try:
        wait_for_api(
            host=args.host,
            port=args.port,
        )

        serve_samples(
            sample_size=args.sample_size,
            random_state=random_state,
            settings=settings,
            reference_date=args.reference_date,
            drop_columns=args.drop_columns,
        )

    finally:
        stop_api_server(api_process)


if __name__ == "__main__":
    main()
