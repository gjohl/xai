from sklearn.metrics import accuracy_score, auc, f1_score, roc_auc_score
import torch

from xai.constants import MODEL_DIR
from xai.data_handlers.utils import load_test_data_mnist_binary
from xai.evaluation_metrics.performance import calculate_evaluation_metrics
from xai.models.simple_cnn import CNNBinaryClassifier


MODEL_FNAME = 'binary_cnn_mnist_run_1.pth'


def run_model_evaluation():
    model = load_binary_classification_model(MODEL_FNAME)
    test_dl = load_test_data_mnist_binary(batch_size=64, shuffle=True)
    output_probs, predicted_classes, test_labels = calculate_model_predictions(model, test_dl)
    metrics = calculate_evaluation_metrics(test_labels, predicted_classes, output_probs)


def load_binary_classification_model(model_filename=MODEL_FNAME):
    model = CNNBinaryClassifier()
    model.load(MODEL_DIR / model_filename)
    return model


def calculate_model_predictions(model, test_dl):
    test_labels_list = []
    output_probs_list = []
    for test_inputs, test_labels in test_dl:
        output_probs = model.probabilities(test_inputs)[:, 1].detach()
        test_labels_list.append(test_labels)
        output_probs_list.append(output_probs)

    test_labels_all = torch.cat(test_labels_list)
    output_probs_all = torch.cat(output_probs_list)
    predicted_classes = (output_probs_all > 0.5) * 1

    return output_probs_all, predicted_classes, test_labels_all
