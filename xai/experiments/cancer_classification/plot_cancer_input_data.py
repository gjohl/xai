import random
import os

from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from xai.constants import FIGURES_DIR, CANCER_DATA_DIR


NUM_INSTANCES = 2
BATCH_SIZE = 128
OUTPUT_FNAME = "cancer_experiment/cancer_images.png"


#############
# Load data #
#############
image_data_dict = {
    'lung': {'n': None, 'aca': None},
    'colon': {'n': None, 'aca': None},
}

for body_part in ('lung', 'colon'):
    for classification in ('n', 'aca'):
        subdirectory = CANCER_DATA_DIR / body_part / 'test' / classification
        fname_list = os.listdir(subdirectory)
        random.shuffle(fname_list)
        full_filepath_list = [subdirectory / fname for fname in fname_list]
        image_data_dict[body_part][classification] = full_filepath_list[:NUM_INSTANCES]

test_data_reshaped = [
    image_data_dict['lung']['n'][0],
    image_data_dict['lung']['aca'][0],
    image_data_dict['colon']['n'][0],
    image_data_dict['colon']['aca'][0],
    image_data_dict['lung']['n'][1],
    image_data_dict['lung']['aca'][1],
    image_data_dict['colon']['n'][1],
    image_data_dict['colon']['aca'][1],
]
test_data_images = [mpimg.imread(image) for image in test_data_reshaped]


#################
# Plot examples #
#################
# Check a single image
plt.imshow(test_data_images[0])

fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, subplot_kw={'xticks': [], 'yticks': []})
for ax, input_data in zip(axs.flat, test_data_images):
    ax.imshow(input_data)

plt.savefig(FIGURES_DIR / OUTPUT_FNAME, format='png')
