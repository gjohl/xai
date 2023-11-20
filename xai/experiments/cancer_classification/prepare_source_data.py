import os
from pathlib import Path
import random
import shutil

from xai.constants import CANCER_DATA_DIR


SOURCE_BASE_DIR = Path("/home/gurp/Documents/phdprep/research_task/lung_colon_image_set/")
NUM_INSTANCES_DICT = {
    'lung': {'train': 4000, 'validation': 500, 'test': 500},
    'colon': {'train': 0, 'validation': 0, 'test': 500},
}


if __name__ == '__main__':
    # Move images to directory structure
    for body_part in ('lung', 'colon'):
        for classification in ('n', 'aca'):
            # Get shuffled filenames per source subfolder
            source_subfolder = SOURCE_BASE_DIR / f'{body_part}_image_sets' / f'{body_part}_{classification}'
            fnames = os.listdir(source_subfolder)
            random.shuffle(fnames)

            start_idx = 0
            for data_category in ('train', 'validation', 'test'):
                # Copy to train, validation or test folder
                num_instances = NUM_INSTANCES_DICT[body_part][data_category]
                fnames_to_copy = fnames[start_idx: start_idx + num_instances]

                if num_instances > 0:
                    print(f"Copying files for {body_part}, {classification}, {data_category}."
                          f" First filename is {fnames_to_copy[0]}")
                    for fname in fnames_to_copy:
                        # Copy files
                        shutil.copyfile(source_subfolder / fname,
                                        CANCER_DATA_DIR / body_part / data_category / classification / fname)
                    start_idx += num_instances
