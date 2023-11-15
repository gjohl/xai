import matplotlib.pyplot as plt
import pandas as pd

from xai.experiments.mnist_binary_classification.evaluate_model_performance import run_and_save_results

digits = (0, 1, 6)
num_samples = 30
output_fname = f"mnist_extrapolation_016_{num_samples}_simplex.pkl"
metrics_dict = run_and_save_results(output_fname, (0, 1, 6), 300)
df = pd.DataFrame(metrics_dict).T

# Plot
def plot_accuracy_distance(df):
    fig, ax1 = plt.subplots()

    color1 = 'tab:red'
    ax1.set_xlabel('Out of distribution proportion')
    ax1.set_ylabel('Accuracy', color=color1)
    ax1.plot(df.index, df['accuracy'], color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color2 = 'tab:blue'
    ax2.set_ylabel('Distance', color=color2)  # we already handled the x-label with ax1
    ax2.plot(df.index, df['simplex'], color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    return fig


df[['accuracy', 'simplex']].plot()
