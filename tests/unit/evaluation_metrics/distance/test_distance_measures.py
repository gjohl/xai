import torch
import pytest

from xai.evaluation_metrics.distance.distance_measures import class_proportions


@pytest.mark.parametrize("labels, expected_zeros_fraction, expected_ones_fraction", [
    [torch.Tensor([0, 0, 0, 0]), 1., 0.],
    [torch.Tensor([1, 1, 1, 1]), 0., 1.],
    [torch.Tensor([0, 0, 1, 1]), 0.5, 0.5],
    [torch.Tensor([0, 1, 1, 1]), 0.25, 0.75],
])
def test_class_proportions(labels, expected_zeros_fraction, expected_ones_fraction):
    zeros_fraction, ones_fraction = class_proportions(labels)
    assert zeros_fraction == expected_zeros_fraction
    assert ones_fraction == expected_ones_fraction
