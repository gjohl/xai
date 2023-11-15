import pickle

from simplexai.explainers.simplex import Simplex
import torch

from xai.constants import MODEL_DIR, RESULTS_DIR
from xai.data_handlers.utils import load_training_data_mnist_binary, load_test_data_mnist_binary
from xai.evaluation_metrics.distance import SimplexDistance, LatentPointwiseDistance, LatentApproxDistance
from xai.evaluation_metrics.performance import calculate_accuracy_metrics
from xai.models.simple_cnn import CNNBinaryClassifier


MODEL_FNAME = 'binary_cnn_mnist_run_1.pth'
BATCH_SIZE = 64


def run_and_save_results(output_fname, digits, num_samples):
    output_fpath = RESULTS_DIR / output_fname
    metrics_dict = run_multiple(digits, num_samples)

    with open(output_fpath, 'wb') as handle:
        pickle.dump(metrics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return metrics_dict


def run_multiple(digits, num_samples):
    model = load_binary_classification_model(MODEL_FNAME)
    train_dl, validation_dl = load_training_data_mnist_binary(batch_size=BATCH_SIZE,
                                                              shuffle=False,
                                                              train_validation_split=[0.8, 0.2])
    source_data, _ = next(iter(train_dl))
    validation_data, _ = next(iter(validation_dl))

    # Fit simplex on the validation data
    source_latents = model.latent_representation(source_data).detach()
    validation_latents = model.latent_representation(validation_data).detach()
    simplex = Simplex(corpus_examples=source_data, corpus_latent_reps=source_latents)
    simplex.fit(test_examples=validation_data, test_latent_reps=validation_latents, reg_factor=0, n_epoch=10000)

    out_of_dist_pct_range = [k/10 for k in range(11)]
    metrics_dict = {}
    for idx, out_of_dist_pct in enumerate(out_of_dist_pct_range):
        print(f"Running metrics for {idx+1} of {len(out_of_dist_pct_range)}")
        results_dict = {}
        count_per_digit = get_count_per_digit(digits, num_samples, out_of_dist_pct)
        test_dl = load_test_data_mnist_binary(batch_size=BATCH_SIZE, shuffle=True,
                                              digits=digits, count_per_digit=count_per_digit)
        target_data, _ = next(iter(test_dl))

        accuracy_metrics = model_accuracy_metrics(model, test_dl)
        distance_metrics = model_distance_metrics(model, source_data, target_data, simplex)
        results_dict = accuracy_metrics | distance_metrics  # Merge dicts into single result
        metrics_dict[out_of_dist_pct] = results_dict

    return metrics_dict


def load_binary_classification_model(model_filename=MODEL_FNAME):
    model = CNNBinaryClassifier()
    model.load(MODEL_DIR / model_filename)
    return model


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
