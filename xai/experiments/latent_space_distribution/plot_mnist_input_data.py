from matplotlib import pyplot as plt
import torchvision

from xai.constants import FIGURES_DIR
from xai.data_handlers.utils import load_test_data_mnist_binary
from xai.data_handlers.mnist import DEFAULT_MNIST_NORMALIZATION
from xai.experiments.latent_space_distribution.plot_utils import get_data_and_labels_for_digits


DIGITS = (0, 1, 2,)
NUM_INSTANCES = 2
BATCH_SIZE = 128
OUTPUT_FNAME = "mnist_digits.png"


#############
# Load data #
#############
# Test data, by original label (digit) rather than boolean, is used for plots
test_dl = load_test_data_mnist_binary(batch_size=BATCH_SIZE,
                                      digits=(0, 1, 2),
                                      count_per_digit={0: 50, 1: 50, 2: 50},
                                      shuffle=False)
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
