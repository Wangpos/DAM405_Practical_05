"""
logger_config.py  —  Task 1: Configure logging
A single place that builds the service logger. It writes to BOTH the console and a
rotating file (service.log) with timestamps and levels.
"""
import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str = "ml-service") -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger() is called more than once
    if logger.handlers:
        return logger

    # Level comes from an env variable: DEBUG in development, INFO in production
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Handler 1: console output
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    # Handler 2: rotating file — max 1 MB per file, keep 3 old backups
    fileh = RotatingFileHandler("service.log", maxBytes=1_000_000, backupCount=3)
    fileh.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(fileh)
    logger.propagate = False  # don't duplicate lines through the root logger
    return logger
