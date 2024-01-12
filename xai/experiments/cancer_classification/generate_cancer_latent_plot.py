from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

from xai.constants import MODEL_DIR, FIGURES_DIR
from xai.models.cancer_classifier_cnn import CancerCNNClassifier
from xai.data_handlers.lung_colon_images import load_cancer_dataset
from xai.evaluation_metrics.distance import SimplexDistance
from xai.experiments.latent_space_distribution.plot_utils import (
    plot_latent_space_2d, plot_latent_space_3d,
    plot_latent_shift, plot_latent_shift_3d,
    plot_latent_space_cancer_2d
)


LATENT_FIGURES_DIR = FIGURES_DIR / 'latent_space'


##############
# Load model #
##############
model_fname = 'cancer_cnn_run_4.pth'
model = CancerCNNClassifier()
model.load(MODEL_DIR / model_fname)


#############
# Load data #
#############
BATCH_SIZE = 64

training_dataset = load_cancer_dataset('lung', 'train')
validation_dataset = load_cancer_dataset('lung', 'validation')
test_dataset_lung = load_cancer_dataset('lung', 'test')
test_dataset_colon = load_cancer_dataset('colon', 'test')

training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_lung_data_loader = DataLoader(test_dataset_lung, batch_size=BATCH_SIZE, shuffle=True)
test_colon_data_loader = DataLoader(test_dataset_colon, batch_size=BATCH_SIZE, shuffle=True)


# Check model works on data
source_data, source_labels = next(iter(training_data_loader))
test_lung_data, test_lung_labels = next(iter(test_lung_data_loader))
test_colon_data, test_colon_labels = next(iter(test_colon_data_loader))
test_data = torch.cat([test_lung_data, test_colon_data])
test_labels = torch.cat([test_lung_labels, test_colon_labels + 2])
# source_data, source_labels = next(iter(training_data_loader))


x = source_data[0]
model.probabilities(x)


#########################################
# Plot different digits in latent space #
#########################################
digits = (0, 1,)

n = 20
latent_plot_fname = "latent_space_scatter_3d.png"

# test_data_digits, labels_digits = get_data_and_labels_for_digits(test_data, labels, digits, n)
latents = model.latent_representation(test_data).detach()
plot_latent_space_cancer_2d(latents[:, [0, 1]], test_labels, digits)
plot_latent_space_cancer_2d(latents[:, [0, 2]], test_labels, digits)
plot_latent_space_cancer_2d(latents[:, [1, 2]], test_labels, digits)

digits = (0, 1, 2, 3)
plot_latent_space_cancer_2d(latents[:, [0, 1]], test_labels, digits)
plot_latent_space_cancer_2d(latents[:, [0, 2]], test_labels, digits)
plot_latent_space_cancer_2d(latents[:, [1, 2]], test_labels, digits)


plot_latent_space_3d(latents, source_labels, digits)
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
