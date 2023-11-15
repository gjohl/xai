import torch

from xai.models.simple_cnn import CNNClassifier, CNNBinaryClassifier


class TestCNNClassifier:
    """Mutliclass CNN."""
    model = CNNClassifier()

    def test_latent_representation(self):
        input_dims = [4, 1, 28, 28]
        mock_input = torch.zeros(input_dims)
        model = self.model
        actual = model.latent_representation(mock_input)
        expected_shape = [input_dims[0], model.fc2.in_features]
        assert list(actual.shape) == expected_shape

    def test_forward(self):
        input_dims = [4, 1, 28, 28]
        mock_input = torch.zeros(input_dims)
        model = self.model
        actual = model.forward(mock_input)
        expected_shape = [input_dims[0], model.fc2.out_features]
        assert list(actual.shape) == expected_shape

    def test_latent_to_presoftmax(self):
        input_dims = [4, 1, 28, 28]
        mock_input = torch.zeros(input_dims)
        model = self.model
        latent = model.latent_representation(mock_input)
        actual = model.latent_to_presoftmax(latent)
        expected_shape = [input_dims[0], model.fc2.out_features]
        assert list(actual.shape) == expected_shape

    def test_presoftmax(self):
        input_dims = [4, 1, 28, 28]
        mock_input = torch.zeros(input_dims)
        model = self.model
        actual = model.presoftmax(mock_input)
        expected_shape = [input_dims[0], model.fc2.out_features]
        assert list(actual.shape) == expected_shape

    def test_probabilties(self):
        input_dims = [4, 1, 28, 28]
        mock_input = torch.zeros(input_dims)
        model = self.model
        actual = model.probabilities(mock_input)
        expected_shape = [input_dims[0], model.fc2.out_features]
        assert list(actual.shape) == expected_shape


class TestCNNBinaryClassifier:
    model = CNNBinaryClassifier()
