# Models
This subdirectory contains the different models used for experiments, as well as code to train the models.

The must be compatible with the Simplex explainer, therefore inherit from the BlackBox abstract base class,
which is itself a child of `torch.nn.module`.
This ensures that they implement `latent_representation` and `forward` methods.
