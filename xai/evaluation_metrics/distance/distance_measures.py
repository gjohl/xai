from typing import Tuple, Dict

import torch

from xai.evaluation_metrics.utils import class_proportions, calculate_norm, DEFAULT_NORM


def calculate_distance_metrics(latent_true: torch.Tensor,
                               latent_approx: torch.Tensor,
                               labels_pred: torch.Tensor,
                               norm: int = DEFAULT_NORM) -> Dict:
    # Residual norm variants
    r_norm = calculate_r_norm(latent_true, latent_approx, vectorwise=False, norm=norm)
    r_vectorwise_norm = calculate_r_norm(latent_true, latent_approx, vectorwise=True, norm=norm)
    r_norm_classwise, r_norm_zeros, r_norm_ones = calculate_r_norm_classwise(
        latent_true, latent_approx, labels_pred, vectorwise=False, norm=norm
    )
    r_vectorwise_norm_classwise, r_vectorwise_norm_zeros, r_vectorwise_norm_ones = calculate_r_norm_classwise(
        latent_true, latent_approx, labels_pred, vectorwise=True, norm=norm
    )
    r_norm_directionwise, r_norm_direction_zeros, r_norm_direction_ones = calculate_r_norm_directionwise(
        latent_true, latent_approx, labels_pred, vectorwise=False, norm=norm
    )
    r_vector_norm_directionwise, r_vector_norm_direction_zeros, r_vector_norm_direction_ones = calculate_r_norm_directionwise(  # noqa
        latent_true, latent_approx, labels_pred, vectorwise=True, norm=norm
    )

    # Latent space norm variants
    h_norm_ratio, h_true_norm, h_approx_norm = calculate_h_norm(latent_true, latent_approx, norm)
    h_norm_classwise, h_norm_zeros_ratio, h_norm_ones_ratio = calculate_h_norm_classwise(latent_true, latent_approx, labels_pred, norm)
    h_norm_directionwise, h_norm_direction_zeros_ratio, h_norm_direction_ones_ratio = calculate_h_norm_directionwise(latent_true, latent_approx, labels_pred, norm)

    # Collect results
    distance_results = {
        # Residual
        'r_norm': r_norm,
        'r_vectorwise_norm': r_vectorwise_norm,
        'r_norm_classwise': r_norm_classwise,
        'r_norm_zeros': r_norm_zeros, 
        'r_norm_ones': r_norm_ones,
        'r_vectorwise_norm_classwise': r_vectorwise_norm_classwise,
        'r_vectorwise_norm_zeros': r_vectorwise_norm_zeros,
        'r_vectorwise_norm_ones': r_vectorwise_norm_ones,
        'r_norm_directionwise': r_norm_directionwise,
        'r_norm_direction_zeros': r_norm_direction_zeros,
        'r_norm_direction_ones': r_norm_direction_ones,
        'r_vector_norm_directionwise': r_vector_norm_directionwise,
        'r_vector_norm_direction_zeros': r_vector_norm_direction_zeros,
        'r_vector_norm_direction_ones': r_vector_norm_direction_ones,
        # Latent space
        'h_norm_ratio': h_norm_ratio,
        'h_true_norm': h_true_norm,
        'h_approx_norm': h_approx_norm,
        'h_norm_classwise': h_norm_classwise,
        'h_norm_zeros_ratio': h_norm_zeros_ratio,
        'h_norm_ones_ratio': h_norm_ones_ratio,
        'h_norm_directionwise': h_norm_directionwise,
        'h_norm_direction_zeros_ratio': h_norm_direction_zeros_ratio,
        'h_norm_direction_ones_ratio': h_norm_direction_ones_ratio,
    }

    return distance_results


def calculate_r_norm(latent_true: torch.Tensor,
                     latent_approx: torch.Tensor,
                     vectorwise: bool = False,
                     norm: int = DEFAULT_NORM) -> float:
    """Calculate the norm of the residual tensor."""
    residual_vectors = latent_true - latent_approx
    return calculate_norm(residual_vectors, vectorwise, norm)


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
    r_norm_zeros = calculate_norm(residual_zeros, vectorwise, norm)
    r_norm_ones = calculate_norm(residual_ones, vectorwise, norm)
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
    # noise_axis_zeros = [k for k in range(num_dims) if k != max_noise_dim_zeros]
    # noise_axis_ones = [k for k in range(num_dims) if k != max_noise_dim_ones]

    # Calculate the class- and direction-specific norms
    zeros_fraction, ones_fraction = class_proportions(labels_pred)
    r_norm_direction_zeros = calculate_norm(residual_zeros[:, max_noise_dim_zeros], vectorwise, norm)
    r_norm_direction_ones = calculate_norm(residual_ones[:, max_noise_dim_ones], vectorwise, norm)
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
    # noise_axis_zeros = [k for k in range(num_dims) if k != max_noise_dim_zeros]
    # noise_axis_ones = [k for k in range(num_dims) if k != max_noise_dim_ones]

    # Calculate the class- and direction-specific norms
    h_true_norm_direction_zeros = float(torch.norm(latent_true_zeros[:, max_noise_dim_zeros], norm))
    h_true_norm_direction_ones = float(torch.norm(latent_true_ones[:, max_noise_dim_ones], norm))
    h_approx_norm_direction_zeros = float(torch.norm(latent_approx_zeros[:, max_noise_dim_zeros], norm))
    h_approx_norm_direction_ones = float(torch.norm(latent_approx_ones[:, max_noise_dim_ones], norm))

    h_norm_direction_zeros_ratio = h_true_norm_direction_zeros / h_approx_norm_direction_zeros
    h_norm_direction_ones_ratio = h_true_norm_direction_ones / h_approx_norm_direction_ones

    # Calculate a weighted mean of the two, weighted by the proportion of instances with that label
    zeros_fraction, ones_fraction = class_proportions(labels_pred)
    h_norm_directionwise = (zeros_fraction * h_norm_direction_zeros_ratio) + (ones_fraction * h_norm_direction_ones_ratio)

    return h_norm_directionwise, h_norm_direction_zeros_ratio, h_norm_direction_ones_ratio
