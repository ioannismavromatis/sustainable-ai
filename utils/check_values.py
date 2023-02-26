from utils import log

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")


def set_time(sampling_rate) -> float:
    if sampling_rate < 0.02:
        custom_logger.warning("Sampling rates below 20Hz are not supported.")
        custom_logger.warning("Using a sampling rate of 20Hz.")
        return 0.02

    return sampling_rate


def set_id(id) -> int:
    if id < 0:
        custom_logger.warning("Running ID cannot be negative.")
        custom_logger.warning("Using a running ID of 0.")
        return 0
    elif not isinstance(id, int):
        custom_logger.warning("Running ID cannot be a float value.")
        custom_logger.warning("Using a running ID of 0.")
        return 0

    return id


def non_negative_int(x) -> int:
    i = int(x)
    if i < 0:
        raise ValueError("Negative values are not allowed")
    return i


def learning_rate(x) -> float:
    i = float(x)
    if i < 0 or i > 1:
        raise ValueError("Wrong value given for learning rate")
    return i
