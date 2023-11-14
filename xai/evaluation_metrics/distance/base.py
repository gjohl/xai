import abc


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
        # TODO GJ: Do we need to pass the model? Or are the model generated latents enough?
        self.model = model
        self.source_data = source_data
        self.target_data = target_data

        # Compute the source and target latent representations using the model
        self.source_latents = self.model.latent_representation(self.source_data).detach()
        self.target_latents = self.model.latent_representation(self.target_data).detach()

        self._distance_per_point = None

    @abc.abstractmethod
    def distance(self):
        """Calculate the distance measure."""
        pass