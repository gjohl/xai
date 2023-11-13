import abc

import torch
from simplexai.explainers.simplex import Simplex


class BaseDistance(abc.ABC):
    def __init__(self, model, source_data, target_data):
        """
        Abstract base class for distance measure calculators.

        All distance measures are a function of: model, source data, target data.

        Parameters
        ----------
        model
            The fitted model.
        source_data
            Data from the source domain, e.g. the training data.
        target_data
            Data from the target domain, e.g. the test data
        """
        self.model = model
        self.source_data = source_data
        self.target_data = target_data

    @abc.abstractmethod
    def distance(self):
        """Calculate the distance measure."""
        pass


class SimplexDistance(BaseDistance):

    def __init__(self, model, source_data, target_data, simplex=None):
        """

        Parameters
        ----------
        model
            The fitted model.
        source_data
            Data from the source domain, e.g. the training data.
        target_data
            Data from the target domain, e.g. the test data
        simplex: simplexai.explainers.simplex.Simplex, optional
            A pre-trained simplex explainer can be passed.
            Default is None, which will train a new simplex model.
        """

        super().__init__(model, source_data, target_data)
        self.simplex = None
        self.source_latents = None
        self.target_latents = None

    def distance(self):
        """float: Simplex distance based on the residual."""
        if self.simplex is None:
            self._fit_simplex()

        target_latents_approx = self.simplex.latent_approx()
        # TODO GJ: scale this? this is sum of squared errors - divide by len(self.target_latents) to make MSE?
        residual = torch.sqrt(torch.sum((self.target_latents - target_latents_approx) ** 2))

        return float(residual)

    def _fit_simplex(self):
        """Fit a simplex explainer to the model and data."""
        # Compute the corpus and test latent representations
        self.source_latents = self.model.latent_representation(self.source_data).detach()
        self.target_latents = self.model.latent_representation(self.target_data).detach()

        # Fit the Simplex explainer
        simplex = Simplex(corpus_examples=self.source_data, corpus_latent_reps=self.source_latents)
        simplex.fit(test_examples=self.target_data, test_latent_reps=self.target_latents, reg_factor=0)
        self.simplex = simplex
