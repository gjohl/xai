from pathlib import Path

import xai


BASE_DIR = Path(xai.__file__).parents[1]
DATA_DIR = BASE_DIR / 'xai' / 'data'
