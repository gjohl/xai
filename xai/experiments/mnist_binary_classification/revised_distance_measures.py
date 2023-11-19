"""
For different values of OOD proportion
1. Create a dict of latent_true, latent_approx, label_true, label_predicted, softmax probabilities
2. Experiment with different combinations and aggregations of these
"""
import pickle

import pandas as pd
from simplexai.explainers.simplex import Simplex
import torch
import torchvision

from xai.constants import MODEL_DIR, RESULTS_DIR
from xai.models.simple_cnn import CNNBinaryClassifier2D, CNNBinaryClassifier3D
from xai.data_handlers.mnist import DEFAULT_MNIST_NORMALIZATION
from xai.data_handlers.utils import load_training_data_mnist_binary, load_test_data_mnist_binary
from xai.experiments.utils import (
    model_accuracy_metrics, model_distance_metrics, get_count_per_digit
)
from xai.experiments.latent_space_distribution.plot_utils import plot_latent_space_2d, plot_latent_space_3d


digits = (0, 1, 2,)
num_samples = 100
BATCH_SIZE = 1024


##############
# Load model #
##############
MODEL_FNAME = "binary_cnn_mnist_2d_run_1.pth"
model = CNNBinaryClassifier2D()
output_fname = f"mnist_extrapolation_{''.join([str(digit) for digit in digits])}_{num_samples}_simplex.pkl"
model.load(MODEL_DIR / MODEL_FNAME)


#############
# Load data #
#############
# Source data is used to train simplex model
train_dl, validation_dl = load_training_data_mnist_binary(batch_size=BATCH_SIZE,
                                                          shuffle=False,
                                                          train_validation_split=[0.8, 0.2])
source_data, source_labels = next(iter(train_dl))
validation_data, validation_labels = next(iter(validation_dl))


############################################################
# Calculate latents and labels for different values of OOD #
############################################################
# Fit simplex on the validation data
source_latents = model.latent_representation(source_data).detach()
validation_latents = model.latent_representation(validation_data).detach()
simplex = Simplex(corpus_examples=source_data, corpus_latent_reps=source_latents)

# Results
latent_true_dict = {}
latent_approx_dict = {}
labels_true_dict = {}
labels_pred_dict = {}
model_prob_dict = {}

out_of_dist_pct_range = [k / 10 for k in range(11)]
for idx, out_of_dist_pct in enumerate(out_of_dist_pct_range):
    print(f"Running metrics for {idx + 1} of {len(out_of_dist_pct_range)}")
    # Load test data
    count_per_digit = get_count_per_digit(digits, num_samples, out_of_dist_pct)
    test_dl = load_test_data_mnist_binary(batch_size=BATCH_SIZE, shuffle=True,
                                          digits=digits, count_per_digit=count_per_digit)
    target_data, target_labels = next(iter(test_dl))
    target_latents = model.latent_representation(target_data).detach()

    # Model predictions
    output_probs = model.probabilities(target_data)[:, 1].detach()
    predicted_classes = (output_probs > 0.5) * 1

    # Collect results
    simplex.fit(test_examples=target_data, test_latent_reps=target_latents, reg_factor=0, n_epoch=10000)

    latent_true_dict[out_of_dist_pct] = model.latent_representation(target_data).detach()
    latent_approx_dict[out_of_dist_pct] = simplex.latent_approx()
    labels_true_dict[out_of_dist_pct] = target_labels
    labels_pred_dict[out_of_dist_pct] = predicted_classes
    model_prob_dict[out_of_dist_pct] = output_probs


#######################################################################
# Experiment with different combinations and aggregations of features #
#######################################################################
# Derived inputs to distance
norm = 2

distance_dict = {}
distance_metrics = ['h_true_norm', 'h_approx_norm',
                    'r_norm', 'r_norm_mean',
                    'r_norm_zeros', 'r_norm_ones',
                    'r_norm_mean_zeros', 'r_norm_mean_ones',
                    'r_norm_direction_zeros', 'r_norm_direction_ones',
                    'r_norm_mean_direction_zeros', 'r_norm_mean_direction_ones',
                    'h_true_norm_zeros', 'h_true_norm_ones',
                    'h_approx_norm_zeros', 'h_approx_norm_ones',
                    'h_true_norm_direction_zeros', 'h_true_norm_direction_ones',
                    'h_approx_norm_direction_zeros', 'h_approx_norm_direction_ones']

for idx, out_of_dist_pct in enumerate(out_of_dist_pct_range):
    print(f"Running distance metrics for {idx + 1} of {len(out_of_dist_pct_range)}")
    distance_dict[out_of_dist_pct] = dict.fromkeys(distance_metrics)

    # Distance inputs
    latent_true = latent_true_dict[out_of_dist_pct]
    latent_approx = latent_approx_dict[out_of_dist_pct]
    labels_true = labels_true_dict[out_of_dist_pct]
    labels_pred = labels_pred_dict[out_of_dist_pct]
    model_prob = model_prob_dict[out_of_dist_pct]
    
    # Distance components
    h_true_norm = float(torch.norm(latent_true, norm))
    h_approx_norm = float(torch.norm(latent_approx, norm))
    residual_vectors = latent_true - latent_approx
    r_norm = float(torch.norm(residual_vectors, norm))
    r_norm_mean = float(torch.mean(torch.norm(residual_vectors, norm, dim=1)))
    
    # Split by class
    latent_true_zeros = latent_true[labels_pred == 0]
    latent_true_ones = latent_true[labels_pred == 1]
    h_true_norm_zeros = float(torch.norm(latent_true_zeros, norm))
    h_true_norm_ones = float(torch.norm(latent_true_ones, norm))
    
    latent_approx_zeros = latent_approx[labels_pred == 0]
    latent_approx_ones = latent_approx[labels_pred == 1]
    h_approx_norm_zeros = float(torch.norm(latent_approx_zeros, norm))
    h_approx_norm_ones = float(torch.norm(latent_approx_ones, norm))

    zeros_norm_ratio = h_true_norm_zeros / h_approx_norm_zeros
    ones_norm_ratio = h_true_norm_ones / h_approx_norm_ones

    residual_zeros = residual_vectors[labels_pred == 0]
    residual_ones = residual_vectors[labels_pred == 1]

    r_norm_zeros = float(torch.norm(residual_zeros, norm))
    r_norm_ones = float(torch.norm(residual_ones, norm))
    # r_norm_class = (r_norm_zeros + r_norm_ones) / 2

    r_norm_mean_zeros = float(torch.mean(torch.norm(residual_zeros, norm, dim=1)))
    r_norm_mean_ones = float(torch.mean(torch.norm(residual_ones, norm, dim=1)))
    # r_norm_mean_class = (r_norm_mean_zeros + r_norm_mean_ones) / 2

    # If zeros are spread along the x-axis, the noise axis is the y-axis.
    num_dims = latent_true_zeros.shape[1]
    max_noise_dim_zeros = int(torch.argmax(torch.std(latent_true_zeros, dim=0)))
    max_noise_dim_ones = int(torch.argmax(torch.std(latent_true_ones, dim=0)))
    noise_axis_zeros = [k for k in range(num_dims) if k != max_noise_dim_zeros]
    noise_axis_ones = [k for k in range(num_dims) if k != max_noise_dim_ones]

    h_true_norm_direction_zeros = float(torch.norm(latent_true_zeros[:, noise_axis_zeros], norm))
    h_true_norm_direction_ones = float(torch.norm(latent_true_ones[:, noise_axis_ones], norm))
    h_approx_norm_direction_zeros = float(torch.norm(latent_approx_zeros[:, noise_axis_zeros], norm))
    h_approx_norm_direction_ones = float(torch.norm(latent_approx_ones[:, noise_axis_ones], norm))

    r_norm_direction_zeros = float(torch.norm(residual_zeros[:, noise_axis_zeros], norm))
    r_norm_direction_ones = float(torch.norm(residual_ones[:, noise_axis_ones], norm))
    r_norm_mean_direction_zeros = float(torch.mean(torch.norm(residual_zeros[:, noise_axis_zeros], norm, dim=1)))
    r_norm_mean_direction_ones = float(torch.mean(torch.norm(residual_ones[:, noise_axis_ones], norm, dim=1)))

    # Collect results
    distance_dict[out_of_dist_pct]['h_true_norm'] = h_true_norm
    distance_dict[out_of_dist_pct]['h_approx_norm'] = h_approx_norm
    distance_dict[out_of_dist_pct]['r_norm'] = r_norm
    distance_dict[out_of_dist_pct]['r_norm_mean'] = r_norm_mean
    distance_dict[out_of_dist_pct]['r_norm_zeros'] = r_norm_zeros
    distance_dict[out_of_dist_pct]['r_norm_ones'] = r_norm_ones
    distance_dict[out_of_dist_pct]['r_norm_mean_zeros'] = r_norm_mean_zeros
    distance_dict[out_of_dist_pct]['r_norm_mean_ones'] = r_norm_mean_ones
    distance_dict[out_of_dist_pct]['r_norm_direction_zeros'] = r_norm_direction_zeros
    distance_dict[out_of_dist_pct]['r_norm_direction_ones'] = r_norm_direction_ones
    distance_dict[out_of_dist_pct]['r_norm_mean_direction_zeros'] = r_norm_mean_direction_zeros
    distance_dict[out_of_dist_pct]['r_norm_mean_direction_ones'] = r_norm_mean_direction_ones
    distance_dict[out_of_dist_pct]['h_true_norm_zeros'] = h_true_norm_zeros
    distance_dict[out_of_dist_pct]['h_true_norm_ones'] = h_true_norm_ones
    distance_dict[out_of_dist_pct]['h_approx_norm_zeros'] = h_approx_norm_zeros
    distance_dict[out_of_dist_pct]['h_approx_norm_ones'] = h_approx_norm_ones
    distance_dict[out_of_dist_pct]['h_true_norm_direction_zeros'] = h_true_norm_direction_zeros
    distance_dict[out_of_dist_pct]['h_true_norm_direction_ones'] = h_true_norm_direction_ones
    distance_dict[out_of_dist_pct]['h_approx_norm_direction_zeros'] = h_approx_norm_direction_zeros
    distance_dict[out_of_dist_pct]['h_approx_norm_direction_ones'] = h_approx_norm_direction_ones


###########################
# Combine distance inputs #
###########################
distance_df = pd.DataFrame(distance_dict).T
distance_df['h_norm_ratio'] = distance_df['h_true_norm'] / distance_df['h_approx_norm']
distance_df['h_norm_zeros_ratio'] = distance_df['h_true_norm_zeros'] / distance_df['h_approx_norm_zeros']
distance_df['h_norm_ones_ratio'] = distance_df['h_true_norm_ones'] / distance_df['h_approx_norm_ones']
distance_df['h_norm_class_mean'] = (distance_df['h_norm_zeros_ratio'] + distance_df['h_norm_ones_ratio']) / 2

distance_df['h_norm_direction_zeros_ratio'] = distance_df['h_true_norm_direction_zeros'] / distance_df['h_approx_norm_direction_zeros']
distance_df['h_norm_direction_ones_ratio'] = distance_df['h_true_norm_direction_ones'] / distance_df['h_approx_norm_direction_ones']
distance_df['h_norm_direction_mean'] = (distance_df['h_norm_direction_zeros_ratio'] + distance_df['h_norm_direction_ones_ratio']) / 2

distance_df['r_norm_class'] = (distance_df['r_norm_zeros'] + distance_df['r_norm_ones']) / 2
distance_df['r_norm_mean_class'] = (distance_df['r_norm_mean_zeros'] + distance_df['r_norm_mean_ones']) / 2
distance_df['r_norm_direction'] = (distance_df['r_norm_direction_zeros'] + distance_df['r_norm_direction_ones']) / 2
distance_df['r_norm_direction_spread'] = distance_df['r_norm_direction_ones'] - distance_df['r_norm_direction_zeros']
distance_df['r_norm_direction_ratio'] = (distance_df['r_norm_direction_ones'] / distance_df['r_norm_direction_zeros']) - 1
distance_df['r_norm_mean_direction'] = (distance_df['r_norm_mean_direction_zeros'] + distance_df['r_norm_mean_direction_ones']) / 2


distance_output_cols = ['h_norm_ratio',
                        'h_norm_zeros_ratio', 'h_norm_ones_ratio',
                        'h_norm_direction_zeros_ratio', 'h_norm_direction_ones_ratio',
                        'h_norm_class_mean', 'h_norm_direction_mean']

# distance_df[distance_output_cols].plot()
distance_df[['r_norm']].plot()
distance_df[['r_norm_mean']].plot()

distance_df[['r_norm_zeros', 'r_norm_ones', 'r_norm_class']].plot()
distance_df[['r_norm_mean_zeros', 'r_norm_mean_ones', 'r_norm_mean_class']].plot()
distance_df[['r_norm_direction_zeros', 'r_norm_direction_ones', 'r_norm_direction', 'r_norm_direction_spread']].plot()
distance_df[['r_norm_mean_direction_zeros', 'r_norm_mean_direction_ones', 'r_norm_mean_direction']].plot()
distance_df[['r_norm_direction_ratio']].plot()

distance_df[['h_norm_ratio']].plot()
distance_df[['h_norm_zeros_ratio', 'h_norm_ones_ratio', 'h_norm_class_mean']].plot()
distance_df[['h_norm_direction_zeros_ratio', 'h_norm_direction_ones_ratio', 'h_norm_direction_mean']].plot()


pd.options.display.max_columns = 10
distance_df[distance_output_cols]

# pickle results
results_fpath = RESULTS_DIR / 'distance_investigation'

objects_to_save = [latent_true_dict,
                   latent_approx_dict,
                   labels_true_dict,
                   labels_pred_dict,
                   model_prob_dict,
                   distance_dict]
fnames = ['latent_true_dict',
          'latent_approx_dict',
          'labels_true_dict',
          'labels_pred_dict',
          'model_prob_dict',
          'distance_dict']

for obj_to_save, fname_to_save in zip(objects_to_save, fnames):
    print(f"Saving: {fname_to_save}")

    with open(results_fpath / fname_to_save, 'wb') as handle:
        pickle.dump(obj_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

