import torch

from xai.evaluation_metrics.distance import SimplexDistance
from xai.evaluation_metrics.distance.distance_measures import calculate_h_norm, calculate_h_norm_classwise, calculate_h_norm_directionwise
from xai.evaluation_metrics.utils import DEFAULT_NORM

from xai.evaluation_metrics.performance import calculate_accuracy_metrics


def model_accuracy_metrics(model, test_dl):
    """Evaluate the model metrics for a given set of test digits."""
    output_probs, predicted_classes, test_labels = calculate_model_predictions(model, test_dl)
    metrics = calculate_accuracy_metrics(test_labels, predicted_classes, output_probs)
    return metrics


def model_distance_metrics(model, source_data, target_data, validation_latents_approx, simplex, norm=DEFAULT_NORM):
    # Metrics between test data and source data
    simplex_dist = SimplexDistance(model, source_data, target_data, simplex)
    test_data_results = simplex_dist.distance_metrics(norm=norm)

    # Model predictions
    output_probs = model.probabilities(target_data)[:, 1].detach()
    labels_pred = (output_probs > 0.5) * 1

    # Latent space norm variants
    num_target_data_instances = target_data.shape[0]
    validation_latents_approx_filtered = validation_latents_approx[:num_target_data_instances]
    h_norm_ratio, h_true_norm, h_approx_norm = calculate_h_norm(simplex_dist.target_latents, validation_latents_approx_filtered, norm)
    h_norm_classwise, h_norm_zeros_ratio, h_norm_ones_ratio = calculate_h_norm_classwise(simplex_dist.target_latents, validation_latents_approx_filtered, labels_pred, norm)
    h_norm_directionwise, h_norm_direction_zeros_ratio, h_norm_direction_ones_ratio = calculate_h_norm_directionwise(simplex_dist.target_latents, validation_latents_approx_filtered, labels_pred, norm)

    validation_data_results = {
        'validation_h_norm_ratio': h_norm_ratio,
        'validation_h_true_norm': h_true_norm,
        'validation_h_approx_norm': h_approx_norm,
        'validation_h_norm_classwise': h_norm_classwise,
        'validation_h_norm_zeros_ratio': h_norm_zeros_ratio,
        'validation_h_norm_ones_ratio': h_norm_ones_ratio,
        'validation_h_norm_directionwise': h_norm_directionwise,
        'validation_h_norm_direction_zeros_ratio': h_norm_direction_zeros_ratio,
        'validation_h_norm_direction_ones_ratio': h_norm_direction_ones_ratio,
    }

    return test_data_results | validation_data_results  # Merge dicts into single result


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
