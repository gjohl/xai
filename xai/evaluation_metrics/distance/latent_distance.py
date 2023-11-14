import torch

from xai.evaluation_metrics.distance.base import BaseDistance


class LatentDistance(BaseDistance):
    """Calculate the model-specific distribution distance between source data and target data in the latent space."""
    def distance(self):
        """float: L2 distance in the latent space."""
        # TODO GJ
        # target_latents_approx = self.simplex.latent_approx()
        # residual = torch.sqrt(torch.sum((self.target_latents - target_latents_approx) ** 2))
        # return float(residual)

        pass


class LatentApproxDistance(BaseDistance):
    """Calculate the model-specific distribution distance between source data and target data in the latent space
    using the centroid of the training data and the standard deviations in each latent space dimension."""
    def distance(self):
        """float: L2 distance in the latent space."""
        sigma, centroid = torch.std_mean(self.source_latents, dim=0)
        # TODO GJ: maybe use the unscaled version?
        #  Set this as a class attribute now so we have it to hand without needing to recalculate anything
        self._distance_per_point = self.target_latents - centroid
        self._distance_per_point_scaled = torch.nan_to_num(self._distance_per_point / sigma)

        residual = torch.sqrt(torch.sum(self._distance_per_point_scaled ** 2))
        return float(residual)
