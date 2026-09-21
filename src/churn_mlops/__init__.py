import argparse
import sys
from collections.abc import Sequence

from .training import TrainingResult, run_training_job

__all__ = ["TrainingResult", "main", "run_training_job"]

__version__ = "0.1.0"


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch the package CLI to the project's installed workflow entry points.

    Args:
        argv: Optional argument list to parse. When omitted, the process argv is
            used.
    """
    parser = argparse.ArgumentParser(
        prog="churn-mlops",
        description=(
            "Train, score, and serve churn prediction models with the local "
            "MLflow workflow."
        ),
        epilog=(
            "Examples:\n"
            "  churn-mlops train --config sample_training_config.yaml\n"
            "  churn-mlops batch-predict --input_csv customer.csv --output_csv predictions.csv\n"
            "  churn-mlops serve --host 127.0.0.1 --port 8000 --reload"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser(
        "train",
        help="Train and evaluate a churn model from a YAML config.",
    )
    train_parser.add_argument(
        "--config", required=True, help="Training config file path."
    )

    batch_parser = subparsers.add_parser(
        "batch-predict",
        help="Generate batch predictions from an input CSV file.",
    )
    batch_parser.add_argument("--input_csv", required=True, help="Input CSV file path.")
    batch_parser.add_argument(
        "--output_csv", required=True, help="Output CSV file path."
    )
    batch_parser.add_argument(
        "--index_col",
        required=False,
        default=None,
        help="Optional index column name for the input CSV.",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the FastAPI prediction service.",
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Hostname for the API server."
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port for the API server."
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for local development.",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        raise SystemExit(0)

    if args.command == "train":
        from .train import main as train_main

        sys.argv = ["churn-mlops.train", "--config", args.config]
        train_main()
        return

    if args.command == "batch-predict":
        from .batch_predict import main as batch_predict_main

        command = [
            "churn-mlops.batch_predict",
            "--input_csv",
            args.input_csv,
            "--output_csv",
            args.output_csv,
        ]
        if args.index_col is not None:
            command.extend(["--index_col", args.index_col])
        sys.argv = command
        batch_predict_main()
        return

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "churn_mlops.serving.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return

    parser.print_help()
    raise SystemExit(1)
