import json
import os

from ptflops import get_model_complexity_info

from utils import log

custom_logger = log.get_logger(__name__)
custom_logger = log.set_level(__name__, "info")

RESULTS_DIR = "./results/"


def model_parameters(net, input_size) -> None:
    """
    Calculates and logs model size and complexity metrics.

    Args:
        net: The model object for which metrics need to be computed.

    Returns:
        None
    """
    param_size = 0
    for param in net.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    model_file = RESULTS_DIR + net.__class__.__name__ + ".txt"

    with open(model_file, "w", encoding="utf-8") as filename:
        macs, trainable_params = get_model_complexity_info(
            net,
            input_size,
            as_strings=False,
            print_per_layer_stat=True,
            verbose=False,
            ost=filename,
        )

    custom_logger.debug(
        f"Model size for model {net.__class__.__name__}: {size_all_mb:.3f}MB"
    )
    custom_logger.debug(f"MACs for model {net.__class__.__name__}: {macs}")
    custom_logger.debug(
        f"Trainable params for model {net.__class__.__name__}: {trainable_params}"
    )

    results = RESULTS_DIR + "model_size.json"
    if os.path.exists(results):
        with open(results, encoding="utf-8") as f:
            model_size_dict = json.load(f)
    else:
        model_size_dict = {}

    model_size_dict.update(
        {
            net.__class__.__name__: {
                "size_mb": round(size_all_mb, 3),
                "parameters": param_size,
                "buffer": buffer_size,
                "macs": macs,
                "trainable_params": trainable_params,
            }
        }
    )

    with open(results, "w", encoding="utf-8") as fp:
        json.dump(model_size_dict, fp, indent=4)
