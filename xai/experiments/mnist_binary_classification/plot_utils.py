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
