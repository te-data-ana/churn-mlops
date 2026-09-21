import logging
import os


def configure_logging(
    level: int | str | None = None,
    *,
    log_file: str | None = None,
) -> None:
    """Configure application logging with a timestamped format.

    The log level can be overridden with an explicit value or the LOG_LEVEL
    environment variable. Repeated calls reset the root logger so the app keeps
    a single, consistent configuration during local runs and tests.
    """
    requested_level = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    if isinstance(requested_level, str):
        requested_level = getattr(logging, requested_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(requested_level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
