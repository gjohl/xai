from matplotlib import pyplot as plt

from xai.constants import RESULTS_DIR, FIGURES_DIR
from xai.experiments.mnist_binary_classification.plot_utils import (
    plot_ood_results_with_error_bars, collate_experiment_results
)


CANCER_RESULTS_DIR = RESULTS_DIR / 'cancer_experiment'
CANCER_FIGURES_DIR = FIGURES_DIR / 'cancer_experiment'


mean_df, std_df = collate_experiment_results(CANCER_RESULTS_DIR)

############################################
# Plot columns of interest with error bars #
############################################
fig = plot_ood_results_with_error_bars(mean_df, std_df, ['accuracy', 'auc', 'probability_mean'], "Classification metrics")
plt.savefig(CANCER_FIGURES_DIR / 'classification_metrics.png', format='png')

fig = plot_ood_results_with_error_bars(mean_df, std_df, ['probability_std'], "Probability standard deviation")
plt.savefig(CANCER_FIGURES_DIR / 'probability_std_dev.png', format='png')

fig = plot_ood_results_with_error_bars(mean_df, std_df, ['r_vectorwise_norm'], "Residual norm")
# fig = plot_ood_results_with_error_bars(mean_df, std_df, ['r_norm'], "Residual norm")
plt.savefig(CANCER_FIGURES_DIR / 'residual_norm.png', format='png')

fig = plot_ood_results_with_error_bars(mean_df, std_df, ['h_true_norm', 'h_approx_norm'], "Latent space norm")
plt.savefig(CANCER_FIGURES_DIR / 'h_norm.png', format='png')

fig = plot_ood_results_with_error_bars(mean_df, std_df, ['validation_h_norm_ratio', 'validation_h_norm_classwise'], "Relative latent space norm")
plt.savefig(CANCER_FIGURES_DIR / 'relative_h_norm.png', format='png')


# TODO GJ: Fix bug and recheck
# Residual directionwise
fig = plot_ood_results_with_error_bars(mean_df, std_df, ['r_norm_directionwise'], "Residual norm in out-of-plane axis")
fig = plot_ood_results_with_error_bars(mean_df, std_df, ['r_vector_norm_directionwise'], "Residual norm in out-of-plane axis")

# Relative norm directionwise
fig = plot_ood_results_with_error_bars(mean_df, std_df, ['validation_h_norm_directionwise'], "Relative latent space norm in out-of-plane axis")
