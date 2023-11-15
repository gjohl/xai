import numpy as np
import torch
from simplexai.explainers.simplex import Simplex

from xai.evaluation_metrics.distance.base import BaseDistance


class SimplexDistance(BaseDistance):

    def __init__(self, model, source_data, target_data, simplex=None):
        """
        Calculate the model-specific distribution distance between source data and target data
        using the Simplex residuals.

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
        self.simplex = simplex

    def distance(self):
        """float: Simplex distance based on the residual."""
        if self.simplex is None:
            self._fit_simplex()

        # TODO GJ: scale this? this is sum of squared errors - divide by len(self.target_latents) to make MSE?
        target_latents_approx = self.simplex.latent_approx()
        # TODO GJ: maybe we want to investigate the distribution of these values?
        #  Set this as a class attribute now so we have it to hand without needing to recalculate anything
        self._distance_per_point = self.target_latents - target_latents_approx
        residual = torch.sqrt(torch.sum(self._distance_per_point ** 2))
        return float(residual) / (np.prod(self._distance_per_point.shape))

    def _fit_simplex(self):
        """Fit a simplex explainer to the model and data."""
        simplex = Simplex(corpus_examples=self.source_data, corpus_latent_reps=self.source_latents)
        simplex.fit(test_examples=self.target_data, test_latent_reps=self.target_latents, reg_factor=0)
        self.simplex = simplex
