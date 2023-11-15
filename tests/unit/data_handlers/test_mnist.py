import pytest
from torch.utils.data import DataLoader
import torchvision

from xai.constants import DATA_DIR
from xai.data_handlers.mnist import load_mnist_binary_dataset, dataset_subset_by_digit, DEFAULT_MNIST_NORMALIZATION


def test_load_mnist_binary():
    batch_size = 4
    count_per_digit = {0: 8, 1: 8}
    dataset = load_mnist_binary_dataset(
        train=True,
        target_digit=1,
        subset_size=None,
        data_dir=DATA_DIR,
        digits=(0, 1),
        count_per_digit=count_per_digit
    )
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dl_inputs, dl_labels = next(iter(dl))

    assert len(dl) == sum(count_per_digit.values()) // batch_size
    assert dl_inputs.shape[0] == batch_size
    assert dl_labels.shape[0] == batch_size


class TestDatasetSubSetByDigit:

    @pytest.mark.parametrize('digits', [(0,), (0, 1,), (0, 1, 2,), (5, 6, 7)])
    def test_subset_of_digits(self, digits):
        dataset = torchvision.datasets.MNIST(
            DATA_DIR,
            train=True,
            download=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*DEFAULT_MNIST_NORMALIZATION),
            ]),
        )
        subset = dataset_subset_by_digit(dataset, digits)
        dl = DataLoader(subset, batch_size=16)
        dl_inputs, dl_labels = next(iter(dl))
        assert all([label in digits for label in dl_labels])

    @pytest.mark.parametrize('digits, count_per_digit', [
        [(0,), {0: 10}],
        [(0, 1), {0: 10, 1: 20}],
        [(0, 1, 2,), {0: 10, 1: 20, 2: 10}],
    ])
    def test_count_per_digit(self, digits, count_per_digit):
        dataset = torchvision.datasets.MNIST(
            DATA_DIR,
            train=True,
            download=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*DEFAULT_MNIST_NORMALIZATION),
            ]),
        )
        subset = dataset_subset_by_digit(dataset, digits, count_per_digit)
        dl = DataLoader(subset, batch_size=40)
        dl_inputs, dl_labels = next(iter(dl))
        assert all([label in digits for label in dl_labels])
        assert all([int(sum(dl_labels == digit)) == count_per_digit[digit] for digit in count_per_digit])
