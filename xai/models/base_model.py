import abc
from pathlib import Path

import torch
from simplexai.models.base import BlackBox


class BaseModel(BlackBox):

    @abc.abstractmethod
    def model_category(self):
        pass

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Generate predictions for the given input data

        Parameters
        ----------
        input_data: torch.Tensor
            Input data to generate predictions for.

        Returns
        -------
        outputs: torch.Tensor
            Output of the model.
        """
        with torch.no_grad():
            outputs = self(input_data)
        return outputs

    def save(self, output_path: Path):
        """Save the model parameters to a file.

        Parameters
        ----------
        output_path: path-like
            Filepath to save the model state to.
        """
        torch.save(self.state_dict(), output_path)

    def load(self, model_path: Path):
        """Load the model parameters from a file.

        Parameters
        ----------
        model_path: path-like
            Filepath to load the model state from.
        """
        self.load_state_dict(torch.load(model_path))
