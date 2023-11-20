from matplotlib import pyplot as plt

from xai.constants import RESULTS_DIR
from xai.experiments.mnist_binary_classification.plot_utils import (
    plot_ood_results_with_error_bars, collate_experiment_results
)


MNIST_RESULTS_DIR = RESULTS_DIR / 'mnist_experiment_2'


mean_df, std_df = collate_experiment_results(MNIST_RESULTS_DIR)


############################################
# Plot columns of interest with error bars #
############################################
fig = plot_ood_results_with_error_bars(mean_df, std_df, ['accuracy', 'auc', 'probability_mean'], "Classification measures")
fig = plot_ood_results_with_error_bars(mean_df, std_df, ['probability_std'], "Probability standard deviation")
fig = plot_ood_results_with_error_bars(mean_df, std_df, ['r_vectorwise_norm', 'r_vector_norm_directionwise'], "Residual norm")
fig = plot_ood_results_with_error_bars(mean_df, std_df, ['h_true_norm', 'h_approx_norm'], "Latent space norm")
fig = plot_ood_results_with_error_bars(mean_df, std_df, ['validation_h_norm_ratio', 'validation_h_norm_classwise'], "Relative latent space norm")
fig = plot_ood_results_with_error_bars(mean_df, std_df, ['validation_h_norm_directionwise'], "Relative latent space norm in out-of-plane axis")
