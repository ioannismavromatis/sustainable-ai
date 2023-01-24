"""_LOGGER extension."""

# pylint: disable=global-statement

import logging
from typing import Optional

# This is the _LOGGER's object. Used as a global variable throughout the file.
_LOGGER: Optional[logging.Logger] = None


def get_logger(name) -> logging.Logger:
    """
    Get the Logger with a given name. If the Logger doesn't exist, create it.

    Args:
        name (string): The custom Logger's name.

    Returns:
        logging.Logger: The custom Logger's object
    """
    global _LOGGER

    if _LOGGER is None:
        _LOGGER = setup_custom_logger(name, "INFO")
        _LOGGER.propagate = False

    return _LOGGER


def set_level(name: str, level=logging.INFO) -> logging.Logger:
    """
    Set up the logging level - based on six different levels provided. This assumes that a Logger
    already exists and sets up the handlers, formatters and logging level.

    Args:
        name (str): The custom Logger's name.
        level (str, optional): The logging level. Defaults to logging.INFO.

    Raises:
        RuntimeError: Raised if the get_logger function returns an empty Logger.
        ValueError: Raised if the logging level provided by the user is not correct.

    Returns:
        logging.Logger: The custom Logger's object
    """
    global _LOGGER
    _LOGGER = logging.getLogger(name)

    if _LOGGER is None:
        raise RuntimeError("get_logger call was made before Logger was setup!")

    levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    if level.upper() not in levels:
        raise ValueError(f"The given value '{level}' is not an expected logging level")

    level = levels[level.upper()]
    _LOGGER.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(_get_formatter())
    handler.setLevel(level)
    handler.propagate = False

    if _LOGGER.hasHandlers():
        _LOGGER.handlers.clear()
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False

    return _LOGGER


def setup_custom_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Create a new Logger. Set up the handlers, formatters and logging level.

    Args:
        name (str): The custom Logger's name.
        level (str, optional): The logging level. Defaults to logging.INFO.

    Raises:
        RuntimeError: Raised if the get_logger function returns an empty Logger.
        ValueError: Raised if the logging level provided by the user is not correct.

    Returns:
        logging.Logger: The custom Logger's object
    """

    global _LOGGER

    if _LOGGER is not None:
        raise RuntimeError(f"Logger {name} is already configured!")

    _LOGGER = logging.getLogger(name)
    _LOGGER.handlers.clear()

    levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    if level.upper() not in levels:
        raise ValueError(f"The given value '{level}' is not an expected logging level")

    level = levels[level.upper()]

    handler = logging.StreamHandler()
    handler.setFormatter(_get_formatter())
    handler.setLevel(level)
    handler.propagate = False

    if _LOGGER.hasHandlers():
        _LOGGER.handlers.clear()
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False  # otherwise root _LOGGER prints things again

    return _LOGGER


def _get_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(filename)s:%(lineno)-3d | %(levelname)s | %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
