from typing import Tuple

import torch

DEFAULT_NORM = 2


def calculate_r_norm(latent_true: torch.Tensor,
                     latent_approx: torch.Tensor,
                     norm: int = DEFAULT_NORM) -> float:
    """Calculate the norm of the residual tensor."""
    residual_vectors = latent_true - latent_approx
    return float(torch.norm(residual_vectors, norm))
    
    
def calculate_r_norm_vectorwise(latent_true: torch.Tensor,
                                latent_approx: torch.Tensor,
                                norm: int = DEFAULT_NORM) -> float:
    """Calculate the norm of each residual vector, then average these over all instances."""
    residual_vectors = latent_true - latent_approx
    return float(torch.mean(torch.norm(residual_vectors, norm, dim=1)))


def calculate_h_norm(latent_true: torch.Tensor,
                     latent_approx: torch.Tensor,
                     norm: int = DEFAULT_NORM) -> float:
    """Calculate the norm of the true latent space and its approximation, then take the ratio of these."""
    h_true_norm = float(torch.norm(latent_true, norm))
    h_approx_norm = float(torch.norm(latent_approx, norm))
    return h_true_norm / h_approx_norm


def calculate_h_norm_classwise(latent_true: torch.Tensor,
                               latent_approx: torch.Tensor,
                               labels_pred: torch.Tensor,
                               norm: int = DEFAULT_NORM) -> Tuple[float, float, float]:
    h_true_norm_zeros = float(torch.norm(latent_true[labels_pred == 0], norm))
    h_true_norm_ones = float(torch.norm(latent_true[labels_pred == 1], norm))
    h_approx_norm_zeros = float(torch.norm(latent_approx[labels_pred == 0], norm))
    h_approx_norm_ones = float(torch.norm(latent_approx[labels_pred == 1], norm))
    
    h_norm_zeros_ratio = h_true_norm_zeros / h_approx_norm_zeros
    h_norm_ones_ratio = h_true_norm_ones / h_approx_norm_ones

    # Calculate a weighted mean of the two, weighted by the proportion of instances with that label
    zeros_fraction = float(sum(labels_pred == 0) / labels_pred.shape[0])
    ones_fraction = float(sum(labels_pred == 1) / labels_pred.shape[0])
    h_norm_classwise = (zeros_fraction * h_norm_zeros_ratio) + (ones_fraction * h_norm_ones_ratio)

    return h_norm_classwise, h_norm_zeros_ratio, h_norm_ones_ratio


def calculate_h_norm_directionwise(latent_true: torch.Tensor,
                                   latent_approx: torch.Tensor,
                                   labels_pred: torch.Tensor,
                                   norm: int = DEFAULT_NORM) -> Tuple[float, float, float]:
    # Split tensors by predicted class
    latent_true_zeros = latent_true[labels_pred == 0]
    latent_true_ones = latent_true[labels_pred == 1]
    latent_approx_zeros = latent_approx[labels_pred == 0]
    latent_approx_ones = latent_approx[labels_pred == 1]

    # Select the noise axis.
    # If zeros are spread along the x-axis, the noise axis is the y-axis.
    # Similar for higher dimensions, if the zeros are spread in the X-Y plane, the noise axis is the z-axis.
    num_dims = latent_true_zeros.shape[1]
    max_noise_dim_zeros = int(torch.argmax(torch.std(latent_true_zeros, dim=0)))
    max_noise_dim_ones = int(torch.argmax(torch.std(latent_true_ones, dim=0)))
    noise_axis_zeros = [k for k in range(num_dims) if k != max_noise_dim_zeros]
    noise_axis_ones = [k for k in range(num_dims) if k != max_noise_dim_ones]

    # Calculate the class- and direction-specific norms
    h_true_norm_direction_zeros = float(torch.norm(latent_true_zeros[:, noise_axis_zeros], norm))
    h_true_norm_direction_ones = float(torch.norm(latent_true_ones[:, noise_axis_ones], norm))
    h_approx_norm_direction_zeros = float(torch.norm(latent_approx_zeros[:, noise_axis_zeros], norm))
    h_approx_norm_direction_ones = float(torch.norm(latent_approx_ones[:, noise_axis_ones], norm))

    h_norm_direction_zeros_ratio = h_true_norm_direction_zeros / h_approx_norm_direction_zeros
    h_norm_direction_ones_ratio = h_true_norm_direction_ones / h_approx_norm_direction_ones

    # Calculate a weighted mean of the two, weighted by the proportion of instances with that label
    zeros_fraction = float(sum(labels_pred == 0) / labels_pred.shape[0])
    ones_fraction = float(sum(labels_pred == 1) / labels_pred.shape[0])
    h_norm_directionwise = (zeros_fraction * h_norm_direction_zeros_ratio) + (ones_fraction * h_norm_direction_ones_ratio)

    return h_norm_directionwise, h_norm_direction_zeros_ratio, h_norm_direction_ones_ratio
