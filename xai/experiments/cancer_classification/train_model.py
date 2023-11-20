import torch
from torch.utils.data import DataLoader

from xai.constants import MODEL_DIR
from xai.data_handlers.lung_colon_images import load_cancer_dataset
from xai.models.cancer_classifier_cnn import CancerCNNClassifier
from xai.models.training.learner import Learner

BATCH_SIZE = 64
model_output_filename = "cancer_cnn_run_2.pth"


#######################
# Load data and model #
#######################
model = CancerCNNClassifier()

training_dataset = load_cancer_dataset('lung', 'train')
validation_dataset = load_cancer_dataset('lung', 'validation')
training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Check model works on data
source_data, source_labels = next(iter(training_data_loader))
x = source_data[0]
model.probabilities(x)


###############
# Train model #
###############
learner = Learner(model, training_data_loader, validation_data_loader, num_epochs=10)
learner.fit()

# Save model
learner.save_model(MODEL_DIR / model_output_filename)


################
# Check output #
################
# Load test data
test_dataset = load_cancer_dataset('lung', 'test')
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data, test_labels = next(iter(test_data_loader))

# Calculate accuracy
output_probs = model.probabilities(test_data)[:, 1].detach()
labels_pred = (output_probs > 0.5) * 1
accuracy = float(torch.sum(labels_pred == test_labels) / len(labels_pred))
print(accuracy)
