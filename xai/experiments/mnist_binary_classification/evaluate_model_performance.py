import torch

from xai.constants import MODEL_DIR
from xai.data_handlers.utils import load_test_data_mnist_binary
from xai.evaluation_metrics.performance import calculate_evaluation_metrics
from xai.models.simple_cnn import CNNBinaryClassifier


MODEL_FNAME = 'binary_cnn_mnist_run_1.pth'


def run_multiple(digits, num_samples):
    model = load_binary_classification_model(MODEL_FNAME)

    out_of_dist_pct_range = [k/10 for k in range(11)]
    metrics_dict = {}
    for idx, out_of_dist_pct in enumerate(out_of_dist_pct_range):
        print(f"Running metrics for {idx+1} of {len(out_of_dist_pct_range)}")
        count_per_digit = get_count_per_digit(digits, num_samples, out_of_dist_pct)
        metrics = evaluate_model_metrics(model, digits, count_per_digit)
        metrics_dict[out_of_dist_pct] = metrics

    return metrics_dict


def load_binary_classification_model(model_filename=MODEL_FNAME):
    model = CNNBinaryClassifier()
    model.load(MODEL_DIR / model_filename)
    return model


def evaluate_model_metrics(model, digits, count_per_digit):
    """Evaluate the model metrics for a given set of test digits."""
    test_dl = load_test_data_mnist_binary(batch_size=64, shuffle=True, digits=digits, count_per_digit=count_per_digit)
    output_probs, predicted_classes, test_labels = calculate_model_predictions(model, test_dl)
    metrics = calculate_evaluation_metrics(test_labels, predicted_classes, output_probs)
    return metrics


def calculate_model_predictions(model, test_dl):
    """Use the model to generate predictions on the test data."""
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


def get_count_per_digit(digits, num_samples_per_class, out_of_dist_pct):
    # The count for 0 and 1 are determined completely by num_samples_per_class and out_of_dist_pct
    out_of_dist_num_samples = num_samples_per_class * out_of_dist_pct
    count_per_digit = {
        0: num_samples_per_class - int(out_of_dist_num_samples),
        1: num_samples_per_class
    }

    # The count of out of dist digits depends on how many there are
    out_of_dist_digits = set(digits).difference({0, 1})
    for digit in out_of_dist_digits:
        count_per_digit[digit] = int(out_of_dist_num_samples / len(out_of_dist_digits))

    return count_per_digit


# TODO GJ: remove
# metrics_dict = run_multiple((0, 1, 6), 30)
# {k: v['auc'] for k,v in metrics_dict.items()}
