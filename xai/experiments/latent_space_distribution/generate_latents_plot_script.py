import torchvision

from xai.constants import MODEL_DIR
from xai.models.simple_cnn import CNNBinaryClassifier2D
from xai.data_handlers.utils import load_training_data_mnist_binary, load_test_data_mnist_binary
from xai.data_handlers.mnist import DEFAULT_MNIST_NORMALIZATION
from xai.evaluation_metrics.distance import SimplexDistance
from xai.experiments.latent_space_distribution.plot_utils import (
    plot_latent_space_2d, plot_latent_shift, get_data_and_labels_for_digits
)


##############
# Load model #
##############
MODEL_FNAME = "binary_cnn_mnist_2d_run_1.pth"
model = CNNBinaryClassifier2D()
model.load(MODEL_DIR / MODEL_FNAME)


#############
# Load data #
#############
BATCH_SIZE = 128

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
test_data = test_data[:, None, :, :]  # A 1 dimension went missing somewhere?
labels = test_dl.dataset.dataset.targets


#########################################
# Plot different digits in latent space #
#########################################
digits = (0, 1, 2)
n = 20

test_data_digits, labels_digits = get_data_and_labels_for_digits(test_data, labels, digits, n)
latents = model.latent_representation(test_data_digits).detach()
plot_latent_space_2d(latents, labels_digits, digits)


#######################
# Plot residual shift #
#######################
sd = SimplexDistance(model, source_data, test_data_digits)
sd.distance()
latents_approx = sd.simplex.latent_approx()
plot_latent_shift(latents, latents_approx, labels_digits, digits, keep_n=n)
