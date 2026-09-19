import argparse
import logging

from churn_mlops.config import TMP_DIR, configure_logging
from churn_mlops.data import load_raw_data, validate_inference_data
from churn_mlops.serving import Predictor, load_model

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--index_col", required=False)

    args = parser.parse_args()

    # load raw data
    df = load_raw_data(file_name=args.input_csv, index_col=args.index_col)
    # validate data contract/schema
    df = validate_inference_data(df)

    predictor = Predictor(load_model())

    logger.info(
        "Starting batch prediction with input file '%s' and index colum '%s':",
        args.input_csv,
        args.index_col,
    )

    df_pred = predictor.predict_batch(df=df)

    df_pred.to_csv(TMP_DIR / args.output_csv)

    logger.info(
        "Batch prediction completed: Output written to file '%s' in directory '%s'.",
        args.output_csv,
        TMP_DIR,
    )

    logger.info("Generated %d predictions.", len(df_pred))

    logger.info(
        "Average churn probability: %.4f",
        df_pred["predicted_probability"].mean(),
    )


if __name__ == "__main__":
    configure_logging()
    main()
