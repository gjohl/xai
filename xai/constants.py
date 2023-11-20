from pathlib import Path

import xai

# Filepath constants
BASE_DIR = Path(xai.__file__).parents[1]
DATA_DIR = BASE_DIR / 'xai' / 'data'
MODEL_DIR = BASE_DIR / 'xai' / 'models' / 'saved_models'
RESULTS_DIR = BASE_DIR / 'xai' / 'experiments' / 'results'
FIGURES_DIR = BASE_DIR / 'xai' / 'experiments' / 'report_figures'

# This is a big dataset so stored outside the repo
CANCER_DATA_DIR = Path("/home/gurp/Documents/phdprep/research_task/selected_dataset/lung_colon_image_set")
