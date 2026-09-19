import logging


def configure_logging() -> None:
    """Configure application logging with a timestamped INFO-level format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
