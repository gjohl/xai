from torch.utils.data import DataLoader

from xai.data_handlers.utils import load_training_data_mnist_binary


def test_load_training_data_mnist_binary():
    batch_size = 4
    shuffle = False
    train_validation_split = [0.8, 0.2]

    train_dl, validation_dl = load_training_data_mnist_binary(batch_size, shuffle, train_validation_split)
    assert isinstance(train_dl, DataLoader)
    assert isinstance(validation_dl, DataLoader)
