import pandas as pd

from xai.constants import MODEL_DIR
from xai.models.simple_cnn import CNNBinaryClassifier
from xai.experiments.mnist_binary_classification.evaluate_model_performance import run_and_save_results
from xai.experiments.mnist_binary_classification.plot_utils import plot_accuracy_distance


########################
# Experiment variables #
########################
digits = (0, 1, 2,)
num_samples = 100

##############
# Load model #
##############
model_fname = 'binary_cnn_mnist_run_1.pth'
model = CNNBinaryClassifier()
output_fname = f"mnist_extrapolation_{''.join([str(digit) for digit in digits])}_{num_samples}_simplex.pkl"
model.load(MODEL_DIR / model_fname)

#########################
# Save and plot results #
#########################
metrics_dict = run_and_save_results(model, output_fname, digits, num_samples)
df = pd.DataFrame(metrics_dict).T
plot_accuracy_distance(df)
