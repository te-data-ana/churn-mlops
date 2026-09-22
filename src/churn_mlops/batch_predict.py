import argparse
import logging
from pathlib import Path

from churn_mlops.config import RAW_DATA_DIR, TMP_DIR, configure_logging
from churn_mlops.data import load_raw_data, validate_inference_data
from churn_mlops.serving import Predictor, load_model

logger = logging.getLogger(__name__)


def run_batch_prediction(
    input_csv: str,
    output_csv: str,
    index_col: str | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    tracking_uri: str | None = None,
    model_name: str | None = None,
    model_alias: str | None = None,
) -> Path:
    """Run validated batch inference and write predictions to a directory.

    Args:
        input_csv: Input CSV filename.
        output_csv: Output CSV filename.
        index_col: Optional input index column.
        input_dir: Directory containing the input CSV.
        output_dir: Directory receiving the prediction CSV.
        tracking_uri: Optional MLflow tracking URI.
        model_name: Optional registered model name.
        model_alias: Optional registered model alias.

    Returns:
        Path to the generated prediction CSV.
    """
    resolved_input_dir = input_dir or RAW_DATA_DIR
    resolved_output_dir = output_dir or TMP_DIR

    try:
        logger.info(
            "Starting batch prediction with input file '%s' and index column '%s'.",
            input_csv,
            index_col,
        )

        df = load_raw_data(
            file_name=input_csv,
            index_col=index_col,
            data_dir=resolved_input_dir,
        )
        df = validate_inference_data(df)

        if tracking_uri is None and model_name is None and model_alias is None:
            loaded_model = load_model()
        else:
            loaded_model = load_model(
                tracking_uri=tracking_uri,
                model_name=model_name,
                model_alias=model_alias,
            )

        predictor = Predictor(loaded_model)
        df_pred = predictor.predict_batch(df=df)

        output_path = resolved_output_dir / output_csv
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        df_pred.to_csv(output_path)

        logger.info(
            "Batch prediction completed: Output written to file '%s'.",
            output_path,
        )
        logger.info("Generated %d predictions.", len(df_pred))
        logger.info(
            "Average churn probability: %.4f",
            df_pred["predicted_probability"].mean(),
        )
        return output_path
    except Exception:
        logger.exception(
            "Batch prediction failed for input file '%s' and output file '%s'.",
            input_csv,
            output_csv,
        )
        raise


def main() -> None:
    """Run validated batch inference and write predictions to temp folder.

    The input and output paths are supplied as command-line arguments. The
    registered serving model is loaded and predictions are generated.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--index_col", required=False)
    parser.add_argument("--input_dir", required=False)

    args = parser.parse_args()

    run_batch_prediction(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        index_col=args.index_col,
        input_dir=Path(args.input_dir) if args.input_dir else None,
    )


if __name__ == "__main__":
    configure_logging()
    main()
