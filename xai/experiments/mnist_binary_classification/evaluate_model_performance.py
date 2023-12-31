import pickle
from simplexai.explainers.simplex import Simplex

from xai.constants import RESULTS_DIR
from xai.data_handlers.utils import load_training_data_mnist_binary, load_test_data_mnist_binary
from xai.evaluation_metrics.utils import DEFAULT_NORM
from xai.experiments.utils import (
    model_accuracy_metrics, model_distance_metrics, get_count_per_digit
)


BATCH_SIZE = 1024


def run_and_save_results(model, output_fname, digits, num_samples):
    output_fpath = RESULTS_DIR / output_fname
    metrics_dict = run_multiple(model, digits, num_samples)

    try:
        with open(output_fpath, 'wb') as handle:
            pickle.dump(metrics_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print("Failed to pickle metrics dict")

    return metrics_dict


def run_multiple(model, digits, num_samples):
    # Load source data
    train_dl, validation_dl = load_training_data_mnist_binary(batch_size=BATCH_SIZE,
                                                              shuffle=False,
                                                              train_validation_split=[0.8, 0.2])
    source_data, source_labels = next(iter(train_dl))
    validation_data, validation_labels = next(iter(validation_dl))

    # Fit simplex on the validation data - used for relative measures
    source_latents = model.latent_representation(source_data).detach()
    validation_latents = model.latent_representation(validation_data).detach()
    simplex = Simplex(corpus_examples=source_data, corpus_latent_reps=source_latents)
    simplex.fit(test_examples=validation_data, test_latent_reps=validation_latents, reg_factor=0, n_epoch=10000)
    validation_latents_approx = simplex.latent_approx()

    # Load varying tests sets, fit a simplex model to each, calculate distance and accuracy metrics
    out_of_dist_pct_range = [k / 20 for k in range(21)]
    metrics_dict = {}
    for idx, out_of_dist_pct in enumerate(out_of_dist_pct_range):
        print(f"Running metrics for {idx+1} of {len(out_of_dist_pct_range)}")
        # Load test data
        count_per_digit = get_count_per_digit(digits, num_samples, out_of_dist_pct)
        test_dl = load_test_data_mnist_binary(batch_size=BATCH_SIZE, shuffle=True,
                                              digits=digits, count_per_digit=count_per_digit)
        target_data, target_labels = next(iter(test_dl))

        # Calculate distance and accuracy metrics
        distance_metrics = model_distance_metrics(model, source_data, target_data, validation_latents_approx, norm=DEFAULT_NORM, simplex=None)
        accuracy_metrics = model_accuracy_metrics(model, test_dl)
        results_dict = accuracy_metrics | distance_metrics  # Merge dicts into single result

        metrics_dict[out_of_dist_pct] = results_dict

    return metrics_dict
