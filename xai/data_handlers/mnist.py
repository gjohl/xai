import torch
from torch.utils.data import DataLoader
import torchvision

from xai.data_handlers.constants import DATA_DIR


DEFAULT_MNIST_NORMALIZATION = (0.1307,), (0.3081,)


def load_mnist(
    batch_size: int,
    train: bool,
    subset_size=None,
    shuffle: bool = True,
    data_dir=DATA_DIR
) -> DataLoader:
    """

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
