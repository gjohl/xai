import pandas as pd

from xai.constants import MODEL_DIR
from xai.models.simple_cnn import CNNBinaryClassifier2D
from xai.experiments.mnist_binary_classification.evaluate_model_performance import run_and_save_results


########################
# Experiment variables #
########################
digits = (0, 1, 2,)
num_samples = 80

##############
# Load model #
##############
model_fname = 'binary_cnn_mnist_2d_run_1.pth'
model = CNNBinaryClassifier2D()
model.load(MODEL_DIR / model_fname)

#########################
# Save and plot results #
#########################
for run_number in range(1, 11):
    print(f"RUN NUMBER {run_number}------------------------------")
    output_fname = f"mnist_experiment_4/{''.join([str(digit) for digit in digits])}_{num_samples}_run_{run_number}.pkl"
    metrics_dict = run_and_save_results(model, output_fname, digits, num_samples)


df = pd.DataFrame(metrics_dict).T
