from typing import List, Tuple

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
        plt.scatter(x_data, y_data, c=COLOR_MAP[digit], alpha=0.5, label=f'Digit {digit}')

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    # plt.title("Input digits in the model's latent space")
    plt.legend()
    return fig


def plot_latent_space_3d(latents: torch.Tensor,
                         labels_all: torch.Tensor,
                         digits: List[int]) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for digit in digits:
        # Plot each label one-by-one
        x_data = latents[labels_all == digit, 0]
        y_data = latents[labels_all == digit, 1]
        z_data = latents[labels_all == digit, 2]
        ax.scatter(x_data, y_data, z_data, c=COLOR_MAP[digit], alpha=0.5, label=f'Digit {digit}')

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    # plt.title("Input digits in the model's latent space")
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
            plt.plot(true_xy[0], true_xy[1], marker='o', color=COLOR_MAP[digit], alpha=0.3)
            plt.plot(approx_xy[0], approx_xy[1], marker='x', color=COLOR_MAP[digit], alpha=0.3)

    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    # plt.title("Input digit's movement under simplex")
    return fig


def plot_latent_shift_3d(latents: torch.Tensor,
                         latents_approx: torch.Tensor,
                         labels_all: torch.Tensor,
                         digits: List[int],
                         keep_n: int = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for digit in digits:
        true_latents_digit = latents[labels_all == digit][:keep_n]
        approx_latents_digit = latents_approx[labels_all == digit][:keep_n]
        for idx in range(len(true_latents_digit)):
            true_xy = true_latents_digit[idx]
            approx_xy = approx_latents_digit[idx]
            # Plot line joining start and end point
            ax.plot([true_xy[0], approx_xy[0]], [true_xy[1], approx_xy[1]], zs=[true_xy[2], approx_xy[2]],
                    c=COLOR_MAP[digit],  linestyle="--", alpha=0.3)
            # Plot different markers for start and ends
            ax.plot(true_xy[0], true_xy[1], zs=true_xy[2], marker='o', color=COLOR_MAP[digit], alpha=0.3)
            ax.plot(approx_xy[0], approx_xy[1], zs=true_xy[2], marker='x', color=COLOR_MAP[digit], alpha=0.3)

    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    # plt.title("Input digit's movement under simplex")
    return fig


def get_data_and_labels_for_digits(data: torch.Tensor,
                                   labels: torch.Tensor,
                                   digits: List[int],
                                   n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a data tensor and a label tensor, each containing n instances, for the given digits."""
    data_list = []
    label_list = []
    for digit in digits:
        digit_mask = get_digit_mask(labels, digit, n)
        data_list.append(data[digit_mask])
        label_list.append(labels[digit_mask])

    data_output = torch.cat(data_list)
    labels_output = torch.cat(label_list)

    return data_output, labels_output


def get_digit_mask(labels: torch.Tensor,
                   digit: int, n: int) -> torch.Tensor:
    """Return a boolean mask for the first n of the given digit."""
    label_mask = labels == digit
    count_mask = torch.cumsum(label_mask, 0) <= n
    idx_mask = label_mask & count_mask
    return idx_mask
