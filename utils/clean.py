import os
import glob

RESULTS_DIRECTORY = './results'
MODELS_DIRECTORY = './model'

def clean_all():
    result_files = glob.glob(RESULTS_DIRECTORY + '/*')
    for f in result_files:
        os.remove(f)
        
    model_files = glob.glob(MODELS_DIRECTORY + '/*')
    for f in model_files:
        os.remove(f)