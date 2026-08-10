import logging
import sys


def setup_logger(name: str = 'energy_forecasting', level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
