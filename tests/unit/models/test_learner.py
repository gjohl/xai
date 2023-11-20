import torch
from torch.nn.modules.loss import CrossEntropyLoss

from xai.data_handlers.utils import load_training_data_mnist_binary
from xai.models.training.learner import Learner, calculate_one_epoch, DEFAULT_OPTIMIZER, DEFAULT_OPTIMIZER_KWARGS
from xai.models.simple_cnn import CNNBinaryClassifier


class TestLearner:

    def test_fit(self):
        # Load data
        batch_size = 64
        shuffle = True
        train_validation_split = [0.8, 0.2]
        num_epochs = 1
        train_dl, validation_dl = load_training_data_mnist_binary(
            batch_size, shuffle, train_validation_split, subset_size=100
        )

        model = CNNBinaryClassifier()
        learn = Learner(model, train_dl, validation_dl, num_epochs, loss_function=CrossEntropyLoss())
        learn.fit()


def test_calculate_one_epoch():
    # Load data
    batch_size = 64
    shuffle = True
    train_validation_split = [0.8, 0.2]
    train_dl, validation_dl = load_training_data_mnist_binary(
        batch_size, shuffle, train_validation_split, subset_size=100
    )

    # Define mode and hyperparameters
    model = CNNBinaryClassifier()
    loss_function = CrossEntropyLoss()
    optimizer = DEFAULT_OPTIMIZER(model.parameters(), **DEFAULT_OPTIMIZER_KWARGS)

    actual = calculate_one_epoch(model, train_dl, loss_function, optimizer)
    assert isinstance(actual, torch.Tensor)
