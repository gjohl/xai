import torch
from simplexai.explainers.simplex import Simplex

from xai.evaluation_metrics.distance.base import BaseDistance
from xai.evaluation_metrics.distance.distance_measures import calculate_distance_metrics
from xai.evaluation_metrics.utils import DEFAULT_NORM


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

    def distance_metrics(self, norm: int = DEFAULT_NORM) -> dict:
        """Calculate various distance metrics"""
        if self.simplex is None:
            self._fit_simplex()

        # Model predictions
        output_probs = self.model.probabilities(self.target_data)[:, 1].detach()
        labels_pred = (output_probs > 0.5) * 1

        # Simplex approximation
        target_latents_approx = self.simplex.latent_approx()

        return calculate_distance_metrics(self.target_latents, target_latents_approx, labels_pred, norm=norm)

    def distance(self):
        """float: Simplex distance based on the residual."""
        if self.simplex is None:
            self._fit_simplex()

        # TODO GJ: scale this? this is sum of squared errors - divide by len(self.target_latents) to make MSE?
        target_latents_approx = self.simplex.latent_approx()
        # TODO GJ: maybe we want to investigate the distribution of these values?
        #  Set this as a class attribute now so we have it to hand without needing to recalculate anything
        self._distance_per_point = self.target_latents - target_latents_approx
        residual = torch.sqrt(torch.sum(self._distance_per_point ** 2))  # TODO GJ: use torch.norm
        return float(residual)  # / (np.prod(self._distance_per_point.shape))

    def _fit_simplex(self):
        """Fit a simplex explainer to the model and data."""
        simplex = Simplex(corpus_examples=self.source_data, corpus_latent_reps=self.source_latents)
        simplex.fit(test_examples=self.target_data, test_latent_reps=self.target_latents, reg_factor=0, n_epoch=10000)
        self.simplex = simplex
