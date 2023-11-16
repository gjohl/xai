from matplotlib import pyplot as plt
import torch
import torchvision

from xai.constants import MODEL_DIR
from xai.models.simple_cnn import CNNBinaryClassifier2D
from xai.data_handlers.utils import load_training_data_mnist_binary, load_test_data_mnist_binary
from xai.data_handlers.mnist import DEFAULT_MNIST_NORMALIZATION
from xai.evaluation_metrics.distance import SimplexDistance

BATCH_SIZE = 64
MODEL_FNAME = "binary_cnn_mnist_2d_run_1.pth"

model = CNNBinaryClassifier2D()
model.load(MODEL_DIR / MODEL_FNAME)


# ---------------Load data---------------
train_dl, validation_dl = load_training_data_mnist_binary(batch_size=BATCH_SIZE,
                                                          digits=(0, 1, 2),
                                                          shuffle=False,
                                                          train_validation_split=[0.8, 0.2])
test_dl = load_test_data_mnist_binary(batch_size=BATCH_SIZE,
                                      digits=(0, 1, 2),
                                      count_per_digit={0: 50, 1: 50, 2: 50},
                                      shuffle=False)
source_data, source_labels = next(iter(train_dl))
validation_data, validation_labels = next(iter(validation_dl))
test_data, test_labels = next(iter(test_dl))

# Load data by original label (digit) rather than boolean
input_data = test_dl.dataset.dataset.data
input_data = torchvision.transforms.Normalize(*DEFAULT_MNIST_NORMALIZATION)(input_data / 255)
input_data = input_data[:, None, :, :]  # A 1 dimension went missing somewhere?
labels = test_dl.dataset.dataset.targets


# ---------------Plot different digits in latent space---------------
digits = (0, 1, 7)
n = 20

def get_digit_mask(labels, digit, n):
    """Return a boolean mask for the first n of the given digit."""
    label_mask = labels == digit
    count_mask = torch.cumsum(label_mask, 0) <= n
    idx_mask = label_mask & count_mask
    return idx_mask

# Select a subset of points to plot
input_data_list = []
label_list = []
for digit in digits:
    digit_mask = get_digit_mask(labels, digit, n)
    input_data_list.append(input_data[digit_mask])
    label_list.append(labels[digit_mask])

input_data_all = torch.cat(input_data_list)
labels_all = torch.cat(label_list)

# Calculate latents
latents = model.latent_representation(input_data_all).detach()

# Create a scatter plot for each unique label
def plot_latent_space_2d(latents, labels_all, digits):
    fig = plt.figure()

    for digit in digits:
        # Plot each label one-by-one
        x_data = latents[labels_all == digit, 0]
        y_data = latents[labels_all == digit, 1]
        plt.scatter(x_data, y_data, alpha=0.5, label=f'Digit {digit}')

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title("Input digits in the model's latent space")
    plt.legend()
    return fig


plot_latent_space_2d(latents, labels_all, digits)


# ---------------Plot residual shift---------------
# Fit simplex model
sd = SimplexDistance(model, source_data, input_data_all)
sd.distance()
latents_approx = sd.simplex.latent_approx()

# Plot line segments / movement
COLOR_MAP = {
    0: 'k',
    1: 'b',
    2: 'r',
    3: 'g',
    4: 'c',
    5: 'm',
    6: 'y',
    7: '#4B2C5E',  # maroon
    8: '#6E6E6E',  # grey
    9: '#E95C0B',  # orange
}

keep_n = 5


def plot_latent_shift(latents, latents_approx, digits, keep_n=None):
    fig = plt.figure()
    for digit in digits:
        true_latents_digit = latents[labels_all == digit][:keep_n]
        approx_latents_digit = latents_approx[labels_all == digit][:keep_n]
        for idx in range(len(true_latents_digit)):
            true_xy = true_latents_digit[idx]
            approx_xy = approx_latents_digit[idx]
            # Plot line joining start and end point
            plt.plot([true_xy[0], approx_xy[0]], [true_xy[1], approx_xy[1]],
                     c=COLOR_MAP[digit],  linestyle="--", alpha=0.3)
            # Plot different markers for start and ends
            plt.plot(true_xy[0], true_xy[1], marker='o', color=COLOR_MAP[digit])
            plt.plot(approx_xy[0], approx_xy[1], marker='x', color=COLOR_MAP[digit])

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title("Input digit's movement under simplex")
    return fig
