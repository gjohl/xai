import os
import pickle

import pandas as pd

from xai.constants import RESULTS_DIR


MNIST_RESULTS_DIR = RESULTS_DIR / 'mnist_experiment_2'
RESULTS_COLS = [
    'accuracy',
    'f1',
    'auc',
    'probability_mean',
    'probability_std',
    'r_norm',
    'r_vectorwise_norm',
    'r_norm_classwise',
    'r_norm_zeros',
    'r_norm_ones',
    'r_vectorwise_norm_classwise',
    'r_vectorwise_norm_zeros',
    'r_vectorwise_norm_ones',
    'r_norm_directionwise',
    'r_norm_direction_zeros',
    'r_norm_direction_ones',
    'r_vector_norm_directionwise',
    'r_vector_norm_direction_zeros',
    'r_vector_norm_direction_ones',
    'h_norm_ratio',
    'h_true_norm',
    'h_approx_norm',
    'h_norm_classwise',
    'h_norm_zeros_ratio',
    'h_norm_ones_ratio',
    'h_norm_directionwise',
    'h_norm_direction_zeros_ratio',
    'h_norm_direction_ones_ratio',
    'validation_h_norm_ratio',
    'validation_h_true_norm',
    'validation_h_approx_norm',
    'validation_h_norm_classwise',
    'validation_h_norm_zeros_ratio',
    'validation_h_norm_ones_ratio',
    'validation_h_norm_directionwise',
    'validation_h_norm_direction_zeros_ratio',
    'validation_h_norm_direction_ones_ratio'
]


#############
# Load data #
#############
results_df_list = []
for result_fname in os.listdir(MNIST_RESULTS_DIR):
    with open(os.path.join(MNIST_RESULTS_DIR, result_fname), 'rb') as handle:
        metrics_dict = pickle.load(handle)
        results_df_list.append(pd.DataFrame(metrics_dict).T)


####################################
# Calc mean and std of each column #
####################################
df = results_df_list[0]

mean_df = pd.DataFrame(index=df.index)
std_df = pd.DataFrame(index=df.index)

for col in RESULTS_COLS:
    col_df = pd.concat([df[col] for df in results_df_list], axis=1)
    mean_df[col] = col_df.mean(axis=1)
    std_df[col] = col_df.std(axis=1)


