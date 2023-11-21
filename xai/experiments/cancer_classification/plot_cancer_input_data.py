from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision

from xai.constants import FIGURES_DIR
from xai.data_handlers.lung_colon_images import load_cancer_dataset
from xai.experiments.latent_space_distribution.plot_utils import get_data_and_labels_for_digits


NUM_INSTANCES = 2
BATCH_SIZE = 128
OUTPUT_FNAME = "cancer_experiment/cancer_images.png"


#############
# Load data #
#############
# Test data is used for plots
load_cancer_dataset('lung', 'test')


test_dl = load_cancer_dataset()
test_data = test_dl.dataset.dataset.data
test_data = torchvision.transforms.Normalize(*DEFAULT_MNIST_NORMALIZATION)(test_data / 255)
test_data = test_data[:, None, :, :]  # A 1 dimension went missing somewhere?
labels = test_dl.dataset.dataset.targets
test_data_digits, test_labels = get_data_and_labels_for_digits(test_data, labels, DIGITS, NUM_INSTANCES)


#################
# Plot examples #
#################
test_data_digits_reshaped = test_data_digits[[0, 2, 4, 1, 3, 5]]
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, subplot_kw={'xticks': [], 'yticks': []})
for ax, input_data in zip(axs.flat, test_data_digits_reshaped):
    ax.imshow(input_data[0]*-1, cmap='gist_gray')

plt.savefig(FIGURES_DIR / OUTPUT_FNAME, format='png')
