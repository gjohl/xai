import pandas as pd

from xai.constants import MODEL_DIR
from xai.models.cancer_classifier_cnn import CancerCNNClassifier
from xai.experiments.cancer_classification.evaluate_model_performance import run_and_save_results


########################
# Experiment variables #
########################
num_samples = 200


##############
# Load model #
##############
model_fname = 'cancer_cnn_run_2.pth'
model = CancerCNNClassifier()
model.load(MODEL_DIR / model_fname)


#########################
# Save and plot results #
#########################
for run_number in range(1, 11):
    print(f"RUN NUMBER {run_number}------------------------------")
    output_fname = f"{num_samples}_run_{run_number}.pkl"
    metrics_dict = run_and_save_results(model, output_fname, num_samples)


df = pd.DataFrame(metrics_dict).T
