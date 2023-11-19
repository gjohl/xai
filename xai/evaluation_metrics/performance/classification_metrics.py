from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch

from xai.evaluation_metrics.utils import class_proportions


def calculate_accuracy_metrics(test_labels, predicted_classes, output_probs):
    confidence_mean, confidence_std = classwise_probability(output_probs, predicted_classes)
    evaluation_results = {
        'accuracy': accuracy_score(test_labels, predicted_classes),
        'f1': f1_score(test_labels, predicted_classes, average='micro'),
        'auc': roc_auc_score(test_labels, output_probs.detach(), multi_class='ovr'),
        'probability_mean': confidence_mean,
        'probability_std': confidence_std,
    }
    return evaluation_results


def classwise_probability(output_probs, predicted_classes):
    """Calculate the mean and standard deviation of probabilities relative to the predicted classification."""
    # Calculate the mean  and std dev of probability per class
    prob_zeros = output_probs[predicted_classes == 0]
    prob_ones = output_probs[predicted_classes == 1]
    confidence_zeros = float(torch.mean(1 - prob_zeros))
    confidence_ones = float(torch.mean(prob_ones))
    confidence_zeros_std = float(torch.std(1 - prob_zeros))
    confidence_ones_std = float(torch.std(prob_ones))

    # Mean weighted by proportion of samples per classification
    zeros_fraction, ones_fraction = class_proportions(labels=predicted_classes)
    confidence_mean = (zeros_fraction * confidence_zeros) + (ones_fraction * confidence_ones)
    confidence_std = (zeros_fraction * confidence_zeros_std) + (ones_fraction * confidence_ones_std)

    return confidence_mean, confidence_std
