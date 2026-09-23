import argparse
import logging

from churn_mlops import run_training_job
from churn_mlops.config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments, run training, and log the resulting ROC AUC."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_name", required=False)

    args = parser.parse_args()

    training_kwargs = {"config_file": args.config}
    if args.experiment_name is not None:
        training_kwargs["experiment_name"] = args.experiment_name

    result = run_training_job(**training_kwargs)

    logger.info(
        "Successfully trained %s model: AUC=%.4f",
        result.classifier_config["model_name"],
        result.metrics["roc_auc"],
    )


if __name__ == "__main__":
    configure_logging()
    main()
