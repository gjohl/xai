import torch
import torch.nn as nn
import torch.nn.functional as F

from xai.models.simple_cnn import CNNClassifier


class CancerCNNClassifier(CNNClassifier):
    # Adapted from https://github.com/vanderschaarlab/Simplex/blob/0af504927122d59dfc1378b73d0292244213e982/src/simplexai/models/image_recognition.py#L8  # noqa: E501
    def __init__(self) -> None:
        """CNN cancer classifier model"""
        super().__init__()
        # Adjust the in_channels of the first convolutional layer to 3
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5, stride=3)
        self.conv2_drop = nn.Dropout2d()

        # Adjust in_features of the first fully connected layer to match the output size from the convolutional layers
        self.fc1 = nn.Linear(1690, 50)
        self.fc2 = nn.Linear(50, 2)

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the latent representation for the example x.

        This is essentially all but the final layer of the neural network.

        Parameters
        ----------
        x: torch.Tensor
            The input example.

        Returns
        -------
        x: torch.Tensor
            The latent representation of the given input.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 1690)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x

    def model_category(self):
        return "CNNCancerClassifier"
