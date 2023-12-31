from matplotlib import pyplot as plt
import torchvision

from xai.constants import MODEL_DIR, FIGURES_DIR
from xai.models.simple_cnn import CNNBinaryClassifier3D
from xai.data_handlers.utils import load_training_data_mnist_binary, load_test_data_mnist_binary
from xai.data_handlers.mnist import DEFAULT_MNIST_NORMALIZATION
from xai.evaluation_metrics.distance import SimplexDistance
from xai.experiments.latent_space_distribution.plot_utils import (
    plot_latent_space_2d, plot_latent_space_3d,
    plot_latent_shift, plot_latent_shift_3d,
    get_data_and_labels_for_digits
)


LATENT_FIGURES_DIR = FIGURES_DIR / 'latent_space'


##############
# Load model #
##############
MODEL_FNAME = "binary_cnn_mnist_3d_run_1.pth"
model = CNNBinaryClassifier3D()
model.load(MODEL_DIR / MODEL_FNAME)


#############
# Load data #
#############
BATCH_SIZE = 64

# Source data is used to train simplex model
train_dl, validation_dl = load_training_data_mnist_binary(batch_size=BATCH_SIZE,
                                                          shuffle=False,
                                                          train_validation_split=[0.8, 0.2])
source_data, source_labels = next(iter(train_dl))

# Test data, by original label (digit) rather than boolean, is used for plots
test_dl = load_test_data_mnist_binary(batch_size=BATCH_SIZE,
                                      digits=(0, 1, 2),
                                      count_per_digit={0: 50, 1: 50, 2: 50},
                                      shuffle=False)
test_data = test_dl.dataset.dataset.data
test_data = torchvision.transforms.Normalize(*DEFAULT_MNIST_NORMALIZATION)(test_data / 255)
test_data = test_data[:, None, :, :]  # A 1 dimension went missing somewhere
labels = test_dl.dataset.dataset.targets


#########################################
# Plot different digits in latent space #
#########################################
digits = (0, 1, 2)
n = 20
latent_plot_fname = "latent_space_scatter_3d.png"

test_data_digits, labels_digits = get_data_and_labels_for_digits(test_data, labels, digits, n)
latents = model.latent_representation(test_data_digits).detach()
# plot_latent_space_2d(latents[:, [0, 1]], labels_digits, digits)
# plot_latent_space_2d(latents[:, [0, 2]], labels_digits, digits)
# plot_latent_space_2d(latents[:, [1, 2]], labels_digits, digits)

plot_latent_space_3d(latents, labels_digits, digits)
plt.savefig(LATENT_FIGURES_DIR / latent_plot_fname, format='png')


#######################
# Plot residual shift #
#######################
residual_shift_plot_fname = "residual_shift_3d.png"

sd = SimplexDistance(model, source_data, test_data_digits)
sd.distance()
latents_approx = sd.simplex.latent_approx()
# plot_latent_shift(latents[:, [0, 1]], latents_approx[:, [0, 1]], labels_digits, digits, keep_n=n)
# plot_latent_shift(latents[:, [0, 2]], latents_approx[:, [0, 2]], labels_digits, digits, keep_n=n)
# plot_latent_shift(latents[:, [1, 2]], latents_approx[:, [1, 2]], labels_digits, digits, keep_n=n)

plot_latent_shift_3d(latents, latents_approx, labels_digits, digits, keep_n=n)
plt.savefig(LATENT_FIGURES_DIR / residual_shift_plot_fname, format='png')


##################
# Plot residuals #
##################
residual_plot_fname = "residuals_3d.png"

residuals = latents - latents_approx
plot_latent_space_3d(residuals, labels_digits, digits)
plt.savefig(LATENT_FIGURES_DIR / residual_plot_fname, format='png')
