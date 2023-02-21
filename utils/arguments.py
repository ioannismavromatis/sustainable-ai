import argparse

from utils import check_values


def arguments():
    parser = argparse.ArgumentParser(
        description="Energy consumption of various state-of-the-art Machine Learning models"
    )
    parser.add_argument(
        "--batch-size",
        type=check_values.non_negative_int,
        default=128,
        metavar="N",
        help="input batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--test-size",
        type=check_values.non_negative_int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate",
        type=check_values.learning_rate,
        default=0.1,
        metavar="N",
        help="learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=check_values.non_negative_int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: %(default)s)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="save current model",
    )
    parser.add_argument(
        "--tool",
        action="store",
        default="",
        choices=["eco2ai", "codecarbon", "carbontracker"],
        metavar="arg",
        help="tool for power consumption evaluation (default: %(default)s)",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False, help="resume existing model"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="delete old results",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_false",
        default=True,
        help="hide progress bar for training and inference",
    )
    parser.add_argument(
        "--run-id",
        type=check_values.non_negative_int,
        default=0,
        metavar="N",
        help="use separate running identifier for multiple experiments",
    )
    parser.add_argument(
        "--no-results",
        action="store_false",
        default=True,
        help="do not store results for training and inference",
    )
    parser.add_argument(
        "--get-stats",
        action="store_true",
        default=False,
        help="get statistics for the GPU and CPU utilisation",
    )
    args = parser.parse_args()

    return args
