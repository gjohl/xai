import torch
from torch.utils.data import Dataset
import torchvision

from xai.constants import CANCER_DATA_DIR


# Swap  labels so 1 means cancer
LABEL_TRANSFORM_MAP = {
    0: 1,
    1: 0,
    # 2: 1,
}


def load_cancer_dataset(body_part: str, data_category: str) -> Dataset:
    """Load a Dataset for the cancer image dataset.

    Parameters
    ----------
    body_part: str
        The body part to load images for. This can be 'lung' or 'colon'.
    data_category: str
        The data split to load. This can be 'train', 'validation' or 'test'.

    Returns
    -------
    Dataset
        The torch Dataset for the given split.
    """
    root = CANCER_DATA_DIR / body_part / data_category
    dataset = torchvision.datasets.ImageFolder(
        root=root,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            #         torchvision.transforms.Normalize(*DEFAULT_MNIST_NORMALIZATION),
        ]),
        target_transform=lambda x: torch.tensor(LABEL_TRANSFORM_MAP[x], dtype=torch.long)
    )
    return dataset
