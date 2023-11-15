from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from xai.constants import DATA_DIR
from xai.data_handlers.mnist import load_mnist_binary_dataset


def load_training_data_mnist_binary(
        batch_size: int, shuffle: bool, train_validation_split: List[int], subset_size: int = None
) -> Tuple[DataLoader, DataLoader]:
    """Load training and validation DataLoaders for a subset containing 0s and 1s."""
    train_input_dataset = load_mnist_binary_dataset(
        train=True,
        subset_size=subset_size,
        data_dir=DATA_DIR,
        target_digit=1,
        digits=[0, 1],
        count_per_digit=None
    )
    train_set, validation_set = torch.utils.data.random_split(train_input_dataset, train_validation_split)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_dl = DataLoader(validation_set, batch_size=batch_size, shuffle=False)  # No need to shuffle validation

    return train_dl, validation_dl


def load_test_data_mnist_binary(
        batch_size: int, shuffle: bool, digits=[0, 1], count_per_digit=None, subset_size: int = None
) -> DataLoader:
    """Load test DataLoader for binary classification experiment."""
    test_dataset = load_mnist_binary_dataset(
        train=True,
        subset_size=subset_size,
        data_dir=DATA_DIR,
        target_digit=1,
        digits=digits,
        count_per_digit=count_per_digit
    )
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_dl
