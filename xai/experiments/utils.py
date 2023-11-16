import torch

from xai.evaluation_metrics.distance import SimplexDistance, LatentPointwiseDistance, LatentApproxDistance
from xai.evaluation_metrics.performance import calculate_accuracy_metrics


def model_accuracy_metrics(model, test_dl):
    """Evaluate the model metrics for a given set of test digits."""
    output_probs, predicted_classes, test_labels = calculate_model_predictions(model, test_dl)
    metrics = calculate_accuracy_metrics(test_labels, predicted_classes, output_probs)
    return metrics


def model_distance_metrics(model, source_data, target_data, simplex):
    distance_dict = {}
    simplex_dist = SimplexDistance(model, source_data, target_data, simplex)
    latent_pw_dist = LatentPointwiseDistance(model, source_data, target_data)
    latent_approx_dist = LatentApproxDistance(model, source_data, target_data)

    distance_dict['simplex'] = simplex_dist.distance()
    distance_dict['latent_pointwise'] = latent_pw_dist.distance()
    distance_dict['latent_approx'] = latent_approx_dist.distance()
    distance_dict['latent_approx_unscaled'] = float(torch.sqrt(torch.sum(latent_approx_dist._distance_per_point ** 2)))

    return distance_dict


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
