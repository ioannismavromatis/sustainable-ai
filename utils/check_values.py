from utils import log

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")


def set_time(sampling_rate: float) -> float:
    """
    Validates and sets the sampling rate.
    If the input rate is below 0.02, sets it to 0.02.

    :param sampling_rate: The desired sampling rate.
    :return: The validated sampling rate.
    """
    if sampling_rate < 0.02:
        custom_logger.warning("Sampling rates below 20Hz are not supported.")
        custom_logger.warning("Using a sampling rate of 20Hz.")
        return 0.02

    return sampling_rate


def set_id(run_id: int) -> int:
    """
    Validates and sets the running ID.
    If the input is negative or not an integer, sets it to 0.

    :param run_id: The desired running ID.
    :return: The validated running ID.
    """
    if not isinstance(run_id, int):
        custom_logger.warning("Running ID cannot be a float value.")
        custom_logger.warning("Using a running ID of 0.")
        return 0
    elif run_id < 0:
        custom_logger.warning("Running ID cannot be negative.")
        custom_logger.warning("Using a running ID of 0.")
        return 0

    return run_id


def non_negative_int(x: int) -> int:
    """
    Validates if the input is a non-negative integer.

    :param x: The input value.
    :return: The validated non-negative integer.
    :raises ValueError: If the input is negative.
    """
    i = int(x)
    if i < 0:
        raise ValueError("Negative values are not allowed")
    return i


def learning_rate(x: float) -> float:
    """
    Validates the learning rate to be between 0 and 1.

    :param x: The input learning rate.
    :return: The validated learning rate.
    :raises ValueError: If the input is not between 0 and 1.
    """
    i = float(x)
    if i < 0 or i > 1:
        raise ValueError("Wrong value given for learning rate")
    return i
