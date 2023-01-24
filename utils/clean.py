import glob
import os

RESULTS_DIRECTORY = "./results"
MODELS_DIRECTORY = "./model"


def clean_all(args):
    result_files = glob.glob(RESULTS_DIRECTORY + "/*")
    for f in result_files:
        os.remove(f)

    if not args.resume:
        model_files = glob.glob(MODELS_DIRECTORY + "/*")
        for f in model_files:
            os.remove(f)
