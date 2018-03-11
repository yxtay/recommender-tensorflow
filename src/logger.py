import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable
import sys

from src.utils import PROJECT_DIR, make_dirs


def get_logger(name: str, log_path: str = str(PROJECT_DIR / "main.log"),
               console: bool = False) -> logging.Logger:
    """
    Simple logging wrapper that returns logger
    configured to log into file and console.

    Args:
        name (str): name of logger
        log_path (str): path of log file
        console (bool): whether to log on console

    Returns:
        logging.Logger: configured logger
    """
    name = Path(sys.argv[0]).name if name == "__main__" else name
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # ensure that logging handlers are not duplicated
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # rotating file handler
    if log_path:
        make_dirs(log_path, isfile=True)
        fh = RotatingFileHandler(log_path,
                                 maxBytes=10 * 2 ** 20,  # 10 MB
                                 backupCount=1)  # 1 backup
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # null handler
    if not (log_path or console):
        logger.addHandler(logging.NullHandler())

    return logger


def float_array_string(arr: Iterable[float]) -> str:
    """
    format array of floats to 4 decimal places

    Args:
        arr: array of floats

    Returns:
        formatted string
    """
    return "[" + ", ".join(["{:.4f}".format(el) for el in arr]) + "]"
