import pandas as pd
from xai.experiments.mnist_binary_classification.evaluate_model_performance import run_and_save_results

digits = (0, 1, 6)
num_samples = 30
output_fname= "mnist_extrapolation_016_300_simplex.pkl"
metrics_dict = run_and_save_results(output_fname, (0, 1, 6), 300)
df = pd.DataFrame(metrics_dict).T
