from typing import Tuple

import torch


DEFAULT_NORM = 2


def class_proportions(labels: torch.Tensor) -> Tuple[float, float]:
    """Calculate the proportion of negative and positive classifications in the labels."""
    zeros_fraction = float(sum(labels == 0) / labels.shape[0])
    ones_fraction = float(sum(labels == 1) / labels.shape[0])
    return zeros_fraction, ones_fraction


def calculate_norm(residual_vectors: torch.Tensor,
                   vectorwise: bool = False,
                   norm: int = DEFAULT_NORM) -> float:
    """Convenience function to switch between norm calculation methods."""
    if vectorwise:
        # Calculate the norm of each residual vector, then average these over all instances
        return float(torch.mean(torch.norm(residual_vectors, norm, dim=1)))
    else:
        # Calculate the norm over the whole tensor
        return float(torch.norm(residual_vectors, norm))
