import os
import sys
from glob import glob


# Get application directory and name of application.
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_NAME = os.path.basename(APP_ROOT_DIR)

# Get environment directory from app root and set to path.
ENVIRONMENT_DIR = os.path.dirname(APP_ROOT_DIR)
if not ENVIRONMENT_DIR in sys.path:
    sys.path.append(ENVIRONMENT_DIR)

# Set pickle file name and path.
PICKLE_FILE_NAME = "state_rwd_action.pkl"
PICKLE_FILE_PATH = os.path.join(ENVIRONMENT_DIR, PICKLE_FILE_NAME)

# Set model directory path.
MODEL_DIR_PATH = os.path.join(ENVIRONMENT_DIR, 'models')
if not os.path.exists(MODEL_DIR_PATH):
	os.makedirs(MODEL_DIR_PATH)

# Set log directory path.
LOG_DIR_PATH = os.path.join(ENVIRONMENT_DIR, 'logs')
if not os.path.exists(LOG_DIR_PATH):
	os.makedirs(LOG_DIR_PATH)

# Get model files.
def get_model_files():
    files = glob(os.path.join(MODEL_DIR_PATH, '', '*.zip'))
    sorted_files = sorted(files, key=lambda x: os.path.getmtime(x))
    return sorted_files
