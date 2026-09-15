import argparse
import logging

from churn_mlops import configure_logging, run_training_job

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    result = run_training_job(config_file=args.config)

    logger.info(
        "Successfully trained %s model: AUC=%.3f",
        result.classifier_config["classifier_name"],
        result.metrics["roc_auc"],
    )


if __name__ == "__main__":
    configure_logging()
    main()
