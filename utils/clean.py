import glob
import os
from argparse import Namespace

RESULTS_DIRECTORY = "./results"
MODELS_DIRECTORY = "./model"


def clean_all(args: Namespace) -> None:
    """
    Delete all files in the results directory.
    If args.resume is False, also delete all files in the models directory.

    :param args: An object with a 'resume' attribute, typically from argparse.
    """
    # Delete files in the results directory
    result_files = glob.glob(os.path.join(RESULTS_DIRECTORY, "*"))
    for file_path in result_files:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error: {file_path} : {e.strerror}")

    # Delete files in the models directory if not in resume mode
    if not args.resume:
        model_files = glob.glob(os.path.join(MODELS_DIRECTORY, "*"))
        for file_path in model_files:
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error: {file_path} : {e.strerror}")
