import click

import numpy as np
import torch

from xai.constants import DATA_DIR
from xai.data_handlers.mnist import load_mnist_binary
from xai.models.simple_cnn import CNNBinaryClassifier

model = CNNBinaryClassifier()


# Load corpus and test inputs
batch_size = 64
# batch_size_test = 1000

corpus_loader = load_mnist(subset_size=8192, train=True, batch_size=batch_size)  # MNIST train loader
test_loader = load_mnist(subset_size=1024, train=False, batch_size=batch_size)  # MNIST test loader
corpus_inputs, corpus_labels = next(iter(corpus_loader)) # A tensor of corpus inputs
test_inputs, test_labels = next(iter(test_loader)) # A set of inputs to explain

def train_model():
    pass


def load_training_data():
    # Load subset of 0s and 1s
    train_data_loader = load_mnist_binary(
        batch_size=64,
        train=True,
        subset_size=None,
        shuffle=True,
        data_dir=DATA_DIR,
        target_digit=1,
        digits=(0, 1),
        count_per_digit=None
    )

    # Spkit into test and validation set
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

