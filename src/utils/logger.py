import logging, sys
from colorlog import ColoredFormatter

_FMT = "%(log_color)s[%(levelname)s] %(message)s"

def get_logger(name: str = "zeno") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # già configurato
    logger.setLevel(logging.INFO)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(
        ColoredFormatter(_FMT, log_colors={
            "INFO": "cyan", "WARNING": "yellow", "ERROR": "red"
        })
    )
    file = logging.FileHandler("logs/zeno.log", encoding="utf-8")
    file.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(stream)
    logger.addHandler(file)
    return logger
