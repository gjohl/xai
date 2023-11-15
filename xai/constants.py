from pathlib import Path

import xai

# Filepath constants
BASE_DIR = Path(xai.__file__).parents[1]
DATA_DIR = BASE_DIR / 'xai' / 'data'
MODEL_DIR = BASE_DIR / 'xai' / 'models' / 'saved_models'
RESULTS_DIR = BASE_DIR / 'xai' / 'experiments'
