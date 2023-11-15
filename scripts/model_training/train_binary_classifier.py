import click

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from xai.constants import DATA_DIR, MODEL_DIR
from xai.data_handlers.mnist import load_mnist_binary_dataset
from xai.models.simple_cnn import CNNBinaryClassifier
from xai.models.training import Learner

model = CNNBinaryClassifier()


# Load corpus and test inputs


# corpus_loader = load_mnist(subset_size=8192, train=True, batch_size=batch_size)  # MNIST train loader
# test_loader = load_mnist(subset_size=1024, train=False, batch_size=batch_size)  # MNIST test loader
# corpus_inputs, corpus_labels = next(iter(corpus_loader)) # A tensor of corpus inputs
# test_inputs, test_labels = next(iter(test_loader)) # A set of inputs to explain

def run_model_training(
        model_filename,
        batch_size=64,
        shuffle=True,
        train_validation_split=[0.8, 0.2],
        num_epochs=20
):
    train_dl, validation_dl = load_training_data(batch_size, shuffle, train_validation_split)
    learn = train_model(train_dl, validation_dl, num_epochs)
    learn.save_model(MODEL_DIR / model_filename)


def train_model(train_dl, validation_dl, num_epochs):
    """Train a model on the given data"""
    learn = Learner(model, train_dl, validation_dl, num_epochs)
    learn.fit()
    return learn


def load_training_data(batch_size, shuffle, train_validation_split):
    """Load training and validation DataLoaders for a subset containing 0s and 1s."""
    train_input_dataset = load_mnist_binary_dataset(
        train=True,
        subset_size=None,
        data_dir=DATA_DIR,
        target_digit=1,
        digits=[0, 1],
        count_per_digit=None
    )
    train_set, validation_set = torch.utils.data.random_split(train_input_dataset, train_validation_split)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_dl = DataLoader(validation_set, batch_size=batch_size, shuffle=False)  # No need to shuffle validation

    return train_dl, validation_dl
