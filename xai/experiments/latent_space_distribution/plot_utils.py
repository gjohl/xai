from typing import List

from matplotlib import pyplot as plt
import torch


COLOR_MAP = {
    0: 'k',
    1: 'b',
    2: 'r',
    3: 'g',
    4: 'c',
    5: 'm',
    6: 'y',
    7: '#4B2C5E',  # maroon
    8: '#6E6E6E',  # grey
    9: '#E95C0B',  # orange
}


def plot_latent_space_2d(latents: torch.Tensor,
                         labels_all: torch.Tensor,
                         digits: List[int]) -> plt.Figure:
    fig = plt.figure()

    for digit in digits:
        # Plot each label one-by-one
        x_data = latents[labels_all == digit, 0]
        y_data = latents[labels_all == digit, 1]
        plt.scatter(x_data, y_data, alpha=0.5, label=f'Digit {digit}')

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title("Input digits in the model's latent space")
    plt.legend()
    return fig


def plot_latent_shift(latents: torch.Tensor,
                      latents_approx: torch.Tensor,
                      labels_all: torch.Tensor,
                      digits: List[int],
                      keep_n: int = None):
    fig = plt.figure()
    for digit in digits:
        true_latents_digit = latents[labels_all == digit][:keep_n]
        approx_latents_digit = latents_approx[labels_all == digit][:keep_n]
        for idx in range(len(true_latents_digit)):
            true_xy = true_latents_digit[idx]
            approx_xy = approx_latents_digit[idx]
            # Plot line joining start and end point
            plt.plot([true_xy[0], approx_xy[0]], [true_xy[1], approx_xy[1]],
                     c=COLOR_MAP[digit],  linestyle="--", alpha=0.3)
            # Plot different markers for start and ends
            plt.plot(true_xy[0], true_xy[1], marker='o', color=COLOR_MAP[digit])
            plt.plot(approx_xy[0], approx_xy[1], marker='x', color=COLOR_MAP[digit])

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title("Input digit's movement under simplex")
    return fig


def get_digit_mask(labels: torch.Tensor,
                   digit: int, n: int) -> torch.Tensor:
    """Return a boolean mask for the first n of the given digit."""
    label_mask = labels == digit
    count_mask = torch.cumsum(label_mask, 0) <= n
    idx_mask = label_mask & count_mask
    return idx_mask
