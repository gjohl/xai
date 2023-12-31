from matplotlib import pyplot as plt

from xai.constants import RESULTS_DIR, FIGURES_DIR
from xai.experiments.mnist_binary_classification.plot_utils import (
    plot_ood_results_with_error_bars, collate_experiment_results
)


MNIST_RESULTS_DIR = RESULTS_DIR / 'mnist_experiment_combined'
MNIST_FIGURES_DIR = FIGURES_DIR / 'mnist_experiment'


mean_df, std_df = collate_experiment_results(MNIST_RESULTS_DIR)


############################################
# Plot columns of interest with error bars #
############################################
if __name__ == '__main__':
    fig = plot_ood_results_with_error_bars(mean_df, std_df, ['accuracy', 'auc', 'probability_mean'],
                                           "Classification metrics")
    plt.savefig(MNIST_FIGURES_DIR / 'classification_metrics.png', format='png')

    fig = plot_ood_results_with_error_bars(mean_df, std_df, ['probability_std'],
                                           "Probability standard deviation")
    plt.savefig(MNIST_FIGURES_DIR / 'probability_std_dev.png', format='png')

    fig = plot_ood_results_with_error_bars(mean_df[['r_vectorwise_norm']], std_df[['r_vectorwise_norm']],
                                           ['r_vectorwise_norm'], "Residual norm", renamed_cols=['r_norm'])
    # fig = plot_ood_results_with_error_bars(mean_df, std_df, ['r_norm'], "Residual norm")
    plt.savefig(MNIST_FIGURES_DIR / 'residual_norm.png', format='png')

    fig = plot_ood_results_with_error_bars(mean_df, std_df, ['h_true_norm', 'h_approx_norm'],
                                           "Latent space norm")
    plt.savefig(MNIST_FIGURES_DIR / 'h_norm.png', format='png')

    fig = plot_ood_results_with_error_bars(mean_df, std_df, ['validation_h_norm_ratio', 'validation_h_norm_classwise'],
                                           "Relative latent space norm", renamed_cols=['rel_h_norm', 'rel_h_norm_classwise'])
    plt.savefig(MNIST_FIGURES_DIR / 'relative_h_norm.png', format='png')

    fig = plot_ood_results_with_error_bars(mean_df, std_df, ['validation_h_norm_directionwise'],
                                           "Relative latent space norm, out-of-plane", renamed_cols=['rel_h_norm_directionwise'])
    plt.savefig(MNIST_FIGURES_DIR / 'relative_h_norm_directionwise.png', format='png')


    # Unused
    fig = plot_ood_results_with_error_bars(mean_df, std_df, ['r_norm_directionwise'], "Residual norm in out-of-plane axis")
    fig = plot_ood_results_with_error_bars(mean_df, std_df, ['r_vector_norm_directionwise'], "Residual norm in out-of-plane axis")
    fig = plot_ood_results_with_error_bars(mean_df, std_df, ['validation_r_norm_directionwise'], "Residual norm in out-of-plane axis")
    fig = plot_ood_results_with_error_bars(mean_df, std_df, ['validation_r_vector_norm_directionwise'], "Residual norm in out-of-plane axis")
