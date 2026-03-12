"""Loguru logging setup for EauDev."""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_path: str | Path, verbose: bool = False) -> None:
    """Configure loguru: file sink always on, stderr only if verbose."""
    log_path = Path(log_path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    # Always write to file
    logger.add(
        str(log_path),
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )
    # Stderr only in verbose mode
    if verbose:
        logger.add(sys.stderr, level="DEBUG", colorize=True)
