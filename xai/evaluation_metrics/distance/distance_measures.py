from typing import Tuple

import torch

DEFAULT_NORM = 2


def calculate_r_norm(latent_true: torch.Tensor,
                     latent_approx: torch.Tensor,
                     vectorwise: bool = False,
                     norm: int = DEFAULT_NORM) -> float:
    """Calculate the norm of the residual tensor."""
    residual_vectors = latent_true - latent_approx
    return calculate_residual_norm(residual_vectors, vectorwise, norm)


def calculate_r_norm_classwise(latent_true: torch.Tensor,
                               latent_approx: torch.Tensor,
                               labels_pred: torch.Tensor,
                               vectorwise: bool = False,
                               norm: int = DEFAULT_NORM) -> float:
    """Calculate residual norm per predicted class."""
    # Split by predicted class
    residual_vectors = latent_true - latent_approx
    residual_zeros = residual_vectors[labels_pred == 0]
    residual_ones = residual_vectors[labels_pred == 1]

    # Calculate norms
    zeros_fraction, ones_fraction = class_proportions(labels_pred)
    r_norm_zeros = calculate_residual_norm(residual_zeros, vectorwise, norm)
    r_norm_ones = calculate_residual_norm(residual_ones, vectorwise, norm)
    r_norm_classwise = (zeros_fraction * r_norm_zeros) + (ones_fraction * r_norm_ones)

    return r_norm_classwise, r_norm_zeros, r_norm_ones


def calculate_r_norm_directionwise(latent_true: torch.Tensor,
                                   latent_approx: torch.Tensor,
                                   labels_pred: torch.Tensor,
                                   vectorwise: bool = False,
                                   norm: int = DEFAULT_NORM) -> Tuple[float, float, float]:
    """Calculate norms of residual tensor per predicted class in the direction of the noise axis."""
    # Split tensors by predicted class
    residual_vectors = latent_true - latent_approx
    residual_zeros = residual_vectors[labels_pred == 0]
    residual_ones = residual_vectors[labels_pred == 1]
    latent_true_zeros = latent_true[labels_pred == 0]
    latent_true_ones = latent_true[labels_pred == 1]

    # Select the noise axis.
    # If zeros are spread along the x-axis, the noise axis is the y-axis.
    # Similar for higher dimensions, if the zeros are spread in the X-Y plane, the noise axis is the z-axis.
    num_dims = latent_true_zeros.shape[1]
    max_noise_dim_zeros = int(torch.argmax(torch.std(latent_true_zeros, dim=0)))
    max_noise_dim_ones = int(torch.argmax(torch.std(latent_true_ones, dim=0)))
    noise_axis_zeros = [k for k in range(num_dims) if k != max_noise_dim_zeros]
    noise_axis_ones = [k for k in range(num_dims) if k != max_noise_dim_ones]

    # Calculate the class- and direction-specific norms
    zeros_fraction, ones_fraction = class_proportions(labels_pred)
    r_norm_direction_zeros = calculate_residual_norm(residual_zeros[:, noise_axis_zeros], vectorwise, norm)
    r_norm_direction_ones = calculate_residual_norm(residual_ones[:, noise_axis_ones], vectorwise, norm)
    r_norm_directionwise = (zeros_fraction * r_norm_direction_zeros) + (ones_fraction * r_norm_direction_ones)

    return r_norm_directionwise, r_norm_direction_zeros, r_norm_direction_ones


def calculate_h_norm(latent_true: torch.Tensor,
                     latent_approx: torch.Tensor,
                     norm: int = DEFAULT_NORM) -> Tuple[float, float, float]:
    """Calculate the norm of the true latent space and its approximation, then take the ratio of these."""
    h_true_norm = float(torch.norm(latent_true, norm))
    h_approx_norm = float(torch.norm(latent_approx, norm))
    h_norm_ratio = h_true_norm / h_approx_norm
    return h_norm_ratio, h_true_norm, h_approx_norm


def calculate_h_norm_classwise(latent_true: torch.Tensor,
                               latent_approx: torch.Tensor,
                               labels_pred: torch.Tensor,
                               norm: int = DEFAULT_NORM) -> Tuple[float, float, float]:
    """Calculate norms of latent space per predicted class."""
    # Split by predicted class
    h_true_norm_zeros = float(torch.norm(latent_true[labels_pred == 0], norm))
    h_true_norm_ones = float(torch.norm(latent_true[labels_pred == 1], norm))
    h_approx_norm_zeros = float(torch.norm(latent_approx[labels_pred == 0], norm))
    h_approx_norm_ones = float(torch.norm(latent_approx[labels_pred == 1], norm))

    # Calculate norms
    zeros_fraction, ones_fraction = class_proportions(labels_pred)
    h_norm_zeros_ratio = h_true_norm_zeros / h_approx_norm_zeros
    h_norm_ones_ratio = h_true_norm_ones / h_approx_norm_ones
    h_norm_classwise = (zeros_fraction * h_norm_zeros_ratio) + (ones_fraction * h_norm_ones_ratio)

    return h_norm_classwise, h_norm_zeros_ratio, h_norm_ones_ratio


def calculate_h_norm_directionwise(latent_true: torch.Tensor,
                                   latent_approx: torch.Tensor,
                                   labels_pred: torch.Tensor,
                                   norm: int = DEFAULT_NORM) -> Tuple[float, float, float]:
    """Calculate norms of latent space per predicted class in the direction of the noise axis."""
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
    zeros_fraction, ones_fraction = class_proportions(labels_pred)
    h_norm_directionwise = (zeros_fraction * h_norm_direction_zeros_ratio) + (ones_fraction * h_norm_direction_ones_ratio)

    return h_norm_directionwise, h_norm_direction_zeros_ratio, h_norm_direction_ones_ratio


#########
# Utils #
#########
def class_proportions(labels: torch.Tensor) -> Tuple[float, float]:
    """Calculate the proportion of negative and positive classifications in the labels."""
    zeros_fraction = float(sum(labels == 0) / labels.shape[0])
    ones_fraction = float(sum(labels == 1) / labels.shape[0])
    return zeros_fraction, ones_fraction


def calculate_residual_norm(residual_vectors: torch.Tensor,
                            vectorwise: bool = False,
                            norm: int = DEFAULT_NORM) -> float:
    """Convenience function to switch between norm calculation methods."""
    if vectorwise:
        # Calculate the norm of each residual vector, then average these over all instances
        return float(torch.mean(torch.norm(residual_vectors, norm, dim=1)))
    else:
        # Calculate the norm over the whole tensor
        return float(torch.norm(residual_vectors, norm))


