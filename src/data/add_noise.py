"""
This file has helper functions used in our experiments with noisy inputs.
"""

import numpy as np
import torch
import logging


def add_noise_to_d(
    d: np.ndarray | torch.Tensor, noise_to_sig_ratio: float
) -> np.ndarray | torch.Tensor:
    """Additive noise of the form
    d_tilde = d + noise_to_sig_ratio * || d ||_2 / sqrt(n) * noise
    where noise is standard normal.
    Note that the noise follows the same profile in (r, s) and (m, h) coordinates due to the normalization.

    Assumes d has at least 3 dimensions. The first dimensions are batch dimensions, and the last 2 are the
    dimensions of the individual samples to which we are adding noise.

    This function checks the type of the input and calls the corresponding function to add noise.
    """
    assert (
        len(d.shape) > 2
    ), "The input must have at least 3 dimensions. It batches over everything but the final 2 dimensions."

    if isinstance(d, np.ndarray):
        return _add_noise_numpy(d, noise_to_sig_ratio)
    elif isinstance(d, torch.Tensor):
        return _add_noise_torch(d, noise_to_sig_ratio)
    else:
        raise TypeError("d must be either a numpy array or a torch tensor.")


def _add_noise_numpy(d: np.ndarray, noise_to_sig_ratio: float) -> np.ndarray:
    """Adds additive random normal noise to the input.
    If d_tilde is the noisy array, we want to have:
    d_tilde = d + noise_to_sig_ratio * || d ||_2 / sqrt(n) * noise
    where noise is standard normal.

    This function will check wheter the input is complex or not, and will add noise accordingly.
    """

    # Computes the norms of the matrices in the last 2 dimensions. Assumes the dimensions coming before that
    # are batch dimensions.
    d_norm = np.linalg.norm(d, axis=(-2, -1), keepdims=True)
    # print("_add_noise_numpy: d_norm.shape", d_norm.shape)
    # print("_add_noise_numpy: d.shape", d.shape)
    noise = np.random.normal(size=d.shape).astype(d.dtype)
    norm_factor = np.sqrt(d.shape[-2] * d.shape[-1])
    if np.iscomplexobj(d):
        logging.debug("Adding complex values to the noise array.")
        noise += 1j * np.random.normal(size=d.shape).astype(d.dtype)
        norm_factor *= np.sqrt(2)
    return d + noise_to_sig_ratio * d_norm / norm_factor * noise


def _add_noise_torch(d: torch.Tensor, noise_to_sig_ratio: float) -> torch.Tensor:
    """Adds additive random normal noise to the input.
    If d_tilde is the noisy array, we want to have:
    d_tilde = d + noise_to_sig_ratio * || d ||_2 / sqrt(n) * noise
    where noise is standard normal.

    This function will check wheter the input is complex or not, and will add noise accordingly.
    """
    d_norm = torch.linalg.norm(d, dim=(-2, -1), keepdim=True)
    noise = torch.randn(d.shape).to(d.dtype)
    norm_factor = torch.sqrt(
        torch.Tensor(
            [
                d.shape[-2] * d.shape[-1],
            ]
        )
    )
    if torch.is_complex(d):
        logging.debug("Adding complex values to the noise array.")
        noise += 1j * torch.randn(d.shape)
        norm_factor *= torch.sqrt(
            torch.Tensor(
                [
                    2.0,
                ]
            )
        )
    return d + noise_to_sig_ratio * d_norm / norm_factor * noise
