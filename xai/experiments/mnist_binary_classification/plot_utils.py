import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd


def plot_accuracy_distance(df: pd.DataFrame) -> plt.Figure:
    """Plot impurity on x-axis, accuracy on left y-axis and distance on right y-axis."""
    fig, ax1 = plt.subplots()

    # Left y-axis
    color1 = 'tab:red'
    ax1.set_xlabel('Out of distribution proportion')
    ax1.set_ylabel('Accuracy', color=color1)
    ax1.plot(df.index, df['accuracy'], color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Right y-axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = 'tab:blue'
    ax2.set_ylabel('Distance', color=color2)  # we already handled the x-label with ax1
    ax2.plot(df.index, df['simplex'], color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return fig


def plot_ood_results_with_error_bars(mean_df: pd.DataFrame,
                                     std_df: pd.DataFrame,
                                     cols: list,
                                     ylabel: str,
                                     renamed_cols: list = None,
                                     legend: bool = True):
    """Plot the given columns against out-of-distribution, with error bars."""
    if renamed_cols:
        col_mapping = {k: v for k, v in zip(cols, renamed_cols)}
        mean_df = mean_df.rename(columns=col_mapping)
        std_df = std_df.rename(columns=col_mapping)
        cols = renamed_cols

    fig = plt.figure()
    for col in cols:
        plt.plot(mean_df.index, mean_df[col], label=col)
        plt.fill_between(mean_df.index, mean_df[col] - std_df[col], mean_df[col] + std_df[col], alpha=0.2)

        plt.xlabel("Out-of-distribution proportion")
        plt.ylabel(ylabel)

    if legend:
        plt.legend()
    fig.tight_layout()
    return fig


def collate_experiment_results(results_dir: str):
    """Calculate the mean and standard deviation of the results DataFrame across all experiment runs."""
    # Load data
    results_df_list = []
    for result_fname in os.listdir(results_dir):
        with open(os.path.join(results_dir, result_fname), 'rb') as handle:
            metrics_dict = pickle.load(handle)
            results_df_list.append(pd.DataFrame(metrics_dict).T)

    # Calc mean and std of each column
    df = results_df_list[0]
    result_cols = df.columns

    mean_df = pd.DataFrame(index=df.index)
    std_df = pd.DataFrame(index=df.index)

    for col in result_cols:
        col_df = pd.concat([df[col] for df in results_df_list], axis=1)
        mean_df[col] = col_df.mean(axis=1)
        std_df[col] = col_df.std(axis=1)

    return mean_df, std_df
