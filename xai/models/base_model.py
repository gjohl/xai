import abc

import torch
from simplexai.models.base import BlackBox


class BaseModel(BlackBox):

    @abc.abstractmethod
    def model_category(self):
        pass

    def save(self, output_path):
        """Save the model parameters to a file.

        Parameters
        ----------
        output_path: path-like
            Filepath to save the model state to.
        """
        torch.save(self.state_dict(), output_path)

    def load(self, model_path):
        """Load the model parameters from a file.

        Parameters
        ----------
        model_path: path-like
            Filepath to load the model state from.
        """
        self.load_state_dict(torch.load(model_path))
