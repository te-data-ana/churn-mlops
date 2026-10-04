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

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=requested_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
