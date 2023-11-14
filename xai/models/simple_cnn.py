import torch
import torch.nn as nn
import torch.nn.functional as F

from xai.models.base_model import BaseModel


class CNNClassifier(BaseModel):
    # Adapted from https://github.com/vanderschaarlab/Simplex/blob/0af504927122d59dfc1378b73d0292244213e982/src/simplexai/models/image_recognition.py#L8  # noqa: E501
    def __init__(self) -> None:
        """CNN binary classifier model"""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def model_category(self):
        return "SimpleCNN"

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
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass for example `x`."""
        x = self.presoftmax(x)
        return F.log_softmax(x, dim=-1)

    def latent_to_presoftmax(self, h: torch.Tensor) -> torch.Tensor:
        """
        Maps a latent representation to a preactivation output.

        Parameters
        ----------
        h: torch.Tensor
            Latent representations.

        Returns
        -------
        torch.Tensor
            Pre-softmax activations
        """
        return self.fc2(h)

    def presoftmax(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the preactivation outputs for the input `x`.

        This is all of the layers except the final softmax layer.

        Parameters
        ----------
        x: torch.Tensor
            Input features

        Returns
        -------
        torch.Tensor
            Pre-softmax activations
        """
        x = self.latent_representation(x)
        return self.latent_to_presoftmax(x)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the class probabilities for the input `x`.

        Parameters
        ----------
        x: torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Class probabilities.
        """
        x = self.presoftmax(x)
        return F.softmax(x, dim=-1)
