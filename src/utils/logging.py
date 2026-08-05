"""Shared logging setup.

Why not print(): print() gives no timestamps, no severity levels, and no way
to redirect output (e.g. to a file per experiment) without touching every
call site. get_logger() gives every module its own named logger
(e.g. "src.inference.sd_generator") so log lines are traceable to their
source, which matters once training/inference logs start interleaving.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
