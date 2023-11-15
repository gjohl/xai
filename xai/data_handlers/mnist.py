import functools
import operator
from typing import List, Dict

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision

from xai.constants import DATA_DIR


DEFAULT_MNIST_NORMALIZATION = (0.1307,), (0.3081,)


def load_mnist(
    batch_size: int,
    train: bool,
    subset_size=None,
    shuffle: bool = True,
    data_dir=DATA_DIR
) -> DataLoader:
    """Load MNIST data.

    Parameters
    ----------
    batch_size: int
        Number of samples per batch to load.
    train: bool
        Whether to create the dataset using the training image set `train-images-idx3-ubyte` or
        the test set `t10k-images-idx3-ubyte`.
    subset_size: int, optional
        If set, this extracts a subset of a given size rather than the full data set.
        Defaults to None, which loads the entire data set.
    shuffle: bool, optional
        Whether to reshuffle the data at every epoch.
    data_dir: path-like, optional
        The directory to download files to.
        This defaults to the `data` subdirectory of the xai project.

    Returns
    -------
    DataLoader
        A DataLoader containing MNIST image data.

    See Also
    --------
    Adapted from https://github.com/vanderschaarlab/Simplex/blob/0af504927122d59dfc1378b73d0292244213e982/src/simplexai/experiments/mnist.py#L43  # noqa E501
    This is mostly unchanged apart from adding a `data_dir` variable.
    """
    dataset = torchvision.datasets.MNIST(
        data_dir,
        train=train,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*DEFAULT_MNIST_NORMALIZATION),
        ])
    )
    if subset_size:
        dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[:subset_size]
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_mnist_binary_dataset(
    train: bool,
    target_digit: int,
    subset_size=None,
    data_dir=DATA_DIR,
    digits: List[int] = None,
    count_per_digit: Dict[int, int] = None
) -> Dataset:
    """Load MNIST data reframed for a binary classification task of distinguishing 1s.

    Parameters
    ----------
    train: bool
        Whether to create the dataset using the training image set `train-images-idx3-ubyte` or
        the test set `t10k-images-idx3-ubyte`.
    target_digit: int
        The digit to use as the target of the binary classification.
        E.g. if target_digit=1, then the dataset will set target labels to True for 1 and False otherwise.
    subset_size: int, optional
        If set, this extracts a subset of a given size rather than the full data set.
        Defaults to None, which loads the entire data set.
    data_dir: path-like, optional
        The directory to download files to.
        This defaults to the `data` subdirectory of the xai project.
    digits: list of int, optional
        Digits to include from the original dataset.
        If None, include all digits.
    count_per_digit: dict, optional
        Keys are the digits, values are the number of instances of that digit that we want.
        If None, include all instances.


    Returns
    -------
    DataLoader
        A DataLoader containing MNIST image data.
    """
    dataset = torchvision.datasets.MNIST(
        data_dir,
        train=train,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*DEFAULT_MNIST_NORMALIZATION),
        ]),
        target_transform=lambda x: x == target_digit
    )

    if digits:
        dataset = dataset_subset_by_digit(dataset, digits, count_per_digit=count_per_digit)

    if subset_size:
        dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[:subset_size]
        )
    return dataset


def dataset_subset_by_digit(dataset: Dataset, digits: List[int], count_per_digit: Dict[int, int] = None):
    """Select subset containing specific digits.

    Parameters
    ----------
    dataset: Dataset
        The Dataset from which we want to extract a subset.
    digits: list of int
        Digits to include from the original dataset.
        If None, include all digits.
    count_per_digit: dict, optional
        Keys are the digits, values are the number of instances of that digit that we want.
        If None, include all instances.

    Returns
    -------
    Dataset
        The required subset of the Dataset.
    """
    digit_mask_list = []
    for digit in digits:
        digit_mask = dataset.targets == digit
        if count_per_digit:
            # If the user has specified a count n for the digit, only include the first n instances.
            running_count = torch.cumsum(digit_mask, 0)
            required_digit_count = count_per_digit[digit]
            running_count_mask = running_count <= required_digit_count
            digit_mask &= running_count_mask
        digit_mask_list.append(digit_mask)

    # Logical or of all masks, i.e. the union of given digits
    digits_union_mask = functools.reduce(operator.or_, digit_mask_list)
    digits_idx = digits_union_mask.nonzero().squeeze()

    return torch.utils.data.Subset(dataset, digits_idx)
