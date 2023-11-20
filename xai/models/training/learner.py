import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn.modules.loss import _Loss, CrossEntropyLoss
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from xai.models.base_model import BaseModel


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEFAULT_LOSS_FUNCTION = CrossEntropyLoss()
DEFAULT_OPTIMIZER = optim.SGD
DEFAULT_OPTIMIZER_KWARGS = dict(lr=0.01, momentum=0.5, weight_decay=0.01)


class Learner:

    def __init__(
            self,
            model: BaseModel,
            train_loader: DataLoader,
            validation_loader: DataLoader,
            num_epochs: int,
            loss_function: _Loss = DEFAULT_LOSS_FUNCTION,
            optimizer_class: Optimizer = DEFAULT_OPTIMIZER,
            optimizer_kwargs: dict = DEFAULT_OPTIMIZER_KWARGS
    ):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_epochs = num_epochs
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

    def fit(self):
        """Fit the model to the training data using the given hyperparameters."""
        train_model(self.model, self.train_loader, self.validation_loader,
                    self.num_epochs, self.loss_function, self.optimizer)

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Generate predictions for the given input data"""
        return self.model.predict(input_data)

    def save_model(self, output_path: Path):
        """Save the trained model parameters to a file.

        Parameters
        ----------
        output_path: path-like
            Filepath to save the model state to.
        """
        self.model.save(output_path)


def train_model(model, train_loader, validation_loader, num_epochs, loss_function, optimizer):
    logger.info(f"Training model {model}")
    training_losses = []
    validation_losses = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        training_loss = calculate_one_epoch(model, train_loader, loss_function, optimizer)
        validation_loss = calculate_validation_loss(model, validation_loader, loss_function)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        print(f"Epoch: {epoch+1} | Training loss: {training_loss} | Validation loss: {validation_loss}")

    logger.info("Model training complete.")
    return (training_losses, validation_losses)


def calculate_one_epoch(model, train_loader, loss_function, optimizer):
    for batch_idx, data in enumerate(train_loader):
        # Unpack inputs and labels from data loader
        inputs, labels = data

        # Zero your learning weight gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch and compute the loss
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

    return loss


def calculate_validation_loss(model, validation_loader, loss_function, debug=False):
    validation_losses = []
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            batch_outputs = model(inputs)
            batch_loss = loss_function(batch_outputs, labels)
            validation_losses.append(batch_loss)

            if debug:
                logger.debug(batch_loss)

    return np.mean(validation_losses)
