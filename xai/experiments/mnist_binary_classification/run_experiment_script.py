import matplotlib.pyplot as plt
import pandas as pd

from xai.experiments.mnist_binary_classification.evaluate_model_performance import run_and_save_results


def plot_accuracy_distance(df):
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


digits = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
num_samples = 100
output_fname = f"mnist_extrapolation_{''.join([str(digit) for digit in digits])}_{num_samples}_1s.pkl"
metrics_dict = run_and_save_results(output_fname, digits, num_samples)
df = pd.DataFrame(metrics_dict).T
plot_accuracy_distance(df)
