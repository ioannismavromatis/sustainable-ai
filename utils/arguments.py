import argparse


def arguments():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 100)",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
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
        "-s",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--evaluation-tool",
        action="store",
        help="Load Tool for power consumption evaluation",
    )
    parser.add_argument(
        "--resume", "-r", action="store_true", default=False, help="Resume from model"
    )
    parser.add_argument(
        "--fresh",
        "-f",
        action="store_true",
        default=False,
        help="Delete all old results",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_false",
        default=True,
        help="Hide the progress bar for training and inference",
    )
    args = parser.parse_args()

    return args
