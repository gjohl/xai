import pickle
from simplexai.explainers.simplex import Simplex
import torch
from torch.utils.data import DataLoader

from xai.constants import RESULTS_DIR
from xai.data_handlers.lung_colon_images import load_cancer_dataset
from xai.evaluation_metrics.performance import calculate_accuracy_metrics
from xai.evaluation_metrics.utils import DEFAULT_NORM
from xai.experiments.utils import model_distance_metrics


BATCH_SIZE = 256


def run_and_save_results(model, output_fname, num_samples):
    output_fpath = RESULTS_DIR / 'cancer_experiment' / output_fname
    metrics_dict = run_multiple(model, num_samples)

    try:
        with open(output_fpath, 'wb') as handle:
            pickle.dump(metrics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print("Failed to pickle metrics dict")

    return metrics_dict


def run_multiple(model, num_samples):
    # Load data
    training_dataset = load_cancer_dataset('lung', 'train')
    validation_dataset = load_cancer_dataset('lung', 'validation')
    test_dataset_lung = load_cancer_dataset('lung', 'test')
    test_dataset_colon = load_cancer_dataset('colon', 'test')

    training_data_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_lung_data_loader = DataLoader(test_dataset_lung, batch_size=BATCH_SIZE, shuffle=True)
    test_colon_data_loader = DataLoader(test_dataset_colon, batch_size=BATCH_SIZE, shuffle=True)

    source_data, source_labels = next(iter(training_data_loader))
    validation_data, validation_labels = next(iter(validation_data_loader))

    # Fit simplex on the validation data - used for relative measures
    source_latents = model.latent_representation(source_data).detach()
    validation_latents = model.latent_representation(validation_data).detach()
    simplex = Simplex(corpus_examples=source_data, corpus_latent_reps=source_latents)
    simplex.fit(test_examples=validation_data, test_latent_reps=validation_latents, reg_factor=0, n_epoch=10000)
    validation_latents_approx = simplex.latent_approx()

    # Load varying tests sets, fit a simplex model to each, calculate distance and accuracy metrics
    out_of_dist_pct_range = [k/10 for k in range(11)]
    metrics_dict = {}
    for idx, out_of_dist_pct in enumerate(out_of_dist_pct_range):
        print(f"Running metrics for {idx+1} of {len(out_of_dist_pct_range)}")
        # Load test data
        test_lung_data, test_lung_labels = next(iter(test_lung_data_loader))
        test_colon_data, test_colon_labels = next(iter(test_colon_data_loader))

        count_per_body_part = get_count_per_body_part(num_samples, out_of_dist_pct)
        target_data, target_labels = combine_test_data(test_lung_data, test_lung_labels,
                                                       test_colon_data, test_colon_labels,
                                                       count_per_body_part)

        # Calculate distance and accuracy metrics
        distance_metrics = model_distance_metrics(model, source_data, target_data, validation_latents_approx,
                                                  norm=DEFAULT_NORM, simplex=None)
        accuracy_metrics = model_accuracy_metrics(model, target_data, target_labels)
        results_dict = accuracy_metrics | distance_metrics  # Merge dicts into single result

        metrics_dict[out_of_dist_pct] = results_dict

    return metrics_dict


def get_count_per_body_part(num_samples, out_of_dist_pct):
    out_of_dist_num_samples = int(num_samples * out_of_dist_pct)
    return {
        'lung': num_samples - out_of_dist_num_samples,
        'colon': out_of_dist_num_samples
    }


def combine_test_data(test_lung_data, test_lung_labels, test_colon_data, test_colon_labels, count_per_body_part):
    target_data = torch.cat([test_lung_data[:count_per_body_part['lung']],
                             test_colon_data[:count_per_body_part['colon']]])
    target_labels = torch.cat([test_lung_labels[:count_per_body_part['lung']],
                               test_colon_labels[:count_per_body_part['colon']]])

    return target_data, target_labels


def model_accuracy_metrics(model, target_data, target_labels):
    """Evaluate the model metrics for a given test set."""
    output_probs = model.probabilities(target_data)[:, 1].detach()
    predicted_classes = (output_probs > 0.5) * 1
    metrics = calculate_accuracy_metrics(target_labels, predicted_classes, output_probs)
    return metrics
