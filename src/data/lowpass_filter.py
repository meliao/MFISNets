"""
Helper functions for applying Gaussian low-pass filters to the data.
"""

import numpy as np
import torch

from typing import Tuple

def _gaussian(rad: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-1 * np.square(rad) / (sigma**2))

def _extend_1d_grid(n_small: int, dx: float) -> np.ndarray:
    """Helper function for get_extended_grid(). Does the following steps:
    1. Find the nearest power of 2 to 3 * n_small
    2. Generate a 1D grid using that number of samples, with the correct spacing.
    """
    target_n_points = 3 * n_small
    two_exp = int(round(np.log2(target_n_points)))
    n_grid = 2**two_exp

    half_n = n_grid // 2
    lim = half_n * dx

    grid = np.linspace(-lim, lim, n_grid, endpoint=False)
    return grid

def get_extended_grid(n_pixels_small: int, dx: float) -> np.ndarray:
    """1. find nearest power of 2.
    2. Generate a 1D grid using that number of samples, with the correct spacing.
    3. Find the zero index.
    4. Roll the 1D grid to put the zero first.
    5. Make a 2D grid.

    Args:
        n_pixels_small (int): Number of pixels on the small grid. The target
        number of pixels for the extended grid is 3 * n_pixels_small
        dx (float): The spacing for the large grid

    Returns:
        np.ndarray: Has shape (N_extended, N_extented, 2)
    """
    samples = _extend_1d_grid(n_pixels_small, dx)

    zero_idx = np.argwhere(samples == 0)[0, 0]

    samples_rolled = np.roll(samples, -zero_idx)

    X, Y = np.meshgrid(samples_rolled, samples_rolled)

    return np.stack((X, Y), axis=-1)


# def setup_lpf(n_pixels_small: int, dx: float, sigma_val: float) -> np.ndarray:
#     extended_grid = get_extended_grid(n_pixels_small, dx)

#     extended_grid_nrms = np.linalg.norm(extended_grid, axis=-1)

#     gaussian_evals = _gaussian(extended_grid_nrms, sigma_val)

#     # Gaussian should integrate to 1
#     gaussian_evals = gaussian_evals / np.sum(gaussian_evals)

#     # This correction is because we're representing convolution on a discrete grid
#     # gaussian_evals = gaussian_evals * (dx**2)

#     g_fft = np.fft.fft2(gaussian_evals)

#     return g_fft


def apply_lpf(arr: np.ndarray, filter_fourier: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        arr (np.ndarray): has shape (N_samples, N_pixels, N_pixels)
        filter_fourier (np.ndarray): has shape (n_large, n_large). Is the output of
        setup_lpf

    Returns:
        np.ndarray: Has shape (N_samples, N_pixels, N_pixels)
    """
    n = arr.shape[-1]
    arr_torch = torch.from_numpy(arr)
    filter_torch = torch.from_numpy(filter_fourier)

    arr_padded = _zero_pad(arr_torch, filter_torch.shape[0])

    arr_fft = torch.fft.fft2(arr_padded)

    # print("arr_fft shape: ", arr_fft.shape)
    # print("filter_torch.shape: ", filter_torch.shape)

    prod = arr_fft * filter_torch
    # prod = torch.einsum("ab,abc->abc", self.G_fft, x_fft)
    out_padded = torch.fft.ifft2(prod)
    out = out_padded[:, :n, :n]
    return out.real.numpy()

def _zero_pad(v: torch.Tensor, n: int) -> torch.Tensor:
    """v has shape (N_samples, N_pixels, N_pixels) and output has shape (N_samples, n, n)"""
    o = torch.zeros((v.shape[0], n, n), dtype=v.dtype)
    o[:, : v.shape[1], : v.shape[2]] = v

    return o

def prep_lpf_from_wavenum(
    nu: float,
    N_points: int,
    use_hwhm: bool = True,
    pad_mode: str = None,
    manual_pad_len: int = None,
) -> Tuple[np.ndarray, float, int]:
    """Builds a Gaussian low-pass filter in Fourier space
    using spatial wavenumber nu corresponding to the number of elements
    in the domain we would expect to be able to resolve
    (e.g. nu=16 would suggest 16 full wavelengths across the domain)

    Args:
        nu (float): number of resolution elements corresponding to the domain
            i.e., the spatial wavenumber of a wave whose wavelength
                is the target minimum resolution length
            nu=1/(target resolution length)
        N_points (int): number of points for the data
        use_hwhm (bool): whether to use the frequency corresponding to nu as the
            half-width half-maximum point of the gaussian low-pass filter
            Otherwise, the frequency will be set as the standard deviation
        pad_mode (str): determine which kind of padding we want
            options include
                "none": performs no padding; good for a periodic boundary condition
                "double": doubles the length of the input; good for a zero-value boundary condition
                "power-of-two": doubles the length of the input and extends to the nearest
                    power of two; intended for zero-value b.c. and potentially faster execution
                "manual": use the pad_len argument instead
        manual_pad_len (int): padding to use if pad_mode="manual"
    Returns:
        glpf (np.ndarray): gaussian low-pass filter, also in frequency space
            length corresponds to the padding mode
        sig (float): standard deviation of the gaussian filter in frequency space
        pad_len (int): extra length expected when padding the input

    Filter usage: apply the gaussian low-pass filter (glpf) as
        np.fft.ifft(glpf * np.fft.fft(grid_data_padded))
        with no need for shifting (already handled)
        though padding would need to be handled
    """
    pad_mode = pad_mode.lower() if pad_mode is not None else "power-of-two"
    if pad_mode == "power-of-two":
        pad_len = 2 ** int(np.ceil(np.log2(2 * N_points))) - N_points
    elif pad_mode == "double":
        pad_len = N_points
    elif pad_mode == "manual":
        pad_len = manual_pad_len
    else:
        pad_len = 0

    # Set the cutoff frequency. Divide by N_points to keep scaling
    # in terms of original domain length
    freq_cutoff = 2 * np.pi * nu / N_points
    # print(f"freq_cutoff: {freq_cutoff}")
    sig = freq_cutoff / (np.sqrt(2 * np.log(2)) if use_hwhm else 1)

    N_padded = N_points + pad_len
    grid_freqs = np.pi * np.linspace(-1, 1, N_padded, endpoint=False)
    glpf = np.fft.ifftshift(np.exp(-0.5 * (grid_freqs / sig) ** 2))
    return glpf, sig, pad_len


# Apply the filter (1D case)
def apply_filter_fourier_1d(
    data: np.ndarray, filter_fourier: np.ndarray, axis: int = 0
) -> np.ndarray:
    """Applies a filter given in Fourier space onto the data (including padding if necessary)

    Args:
        data (np.ndarray): data array on which to apply the filters. can be >1D
        filter_fourier (np.ndarray): convolution filter in the fourier domain
            (intended to be a gaussian low-pass filter)
            expected to be the same length as the padded data
        axis (int): in case data is multi-dimensional, only apply the filter onto one axis
    Returns:
        result (np.ndarray): result of the convolution in fourier space; preserves data.shape
    """
    # Calculate padding amount from filter shape; also pad the data
    if axis >= data.ndim:
        # Allow negative axis indices but want to catch cases where it's too large
        raise ValueError(
            f"(apply_filter_fourier_1d) encountered axis={axis} "
            f"but data array only has {data.ndim} dimensions"
        )
    axis = axis % data.ndim  # wrap around in case of negative numbering

    pad_len = filter_fourier.shape[0] - data.shape[axis]
    extra_data_shape = list(data.shape)
    extra_data_shape[axis] = pad_len
    data_padded = np.concatenate([data, np.zeros(extra_data_shape)], axis=axis)

    # Apply filter in fourier space (and broadcast the filter shape to match)
    # then go back to the spatial domain
    data_padded_fourier = np.fft.fft(data_padded, axis=axis)
    broadcast_idcs = tuple(
        [slice(None) if i == axis else np.newaxis for i in range(data.ndim)]
    )
    res_padded_fourier = filter_fourier[broadcast_idcs] * data_padded_fourier
    res_padded = np.real(np.fft.ifft(res_padded_fourier, axis=axis))

    unpad_slice = tuple(
        [slice(0, -pad_len) if i == axis else slice(None) for i in range(data.ndim)]
    )
    result = res_padded[unpad_slice]
    return result


# Apply the filter (2D case)
def apply_filter_fourier_2d(
    data: np.ndarray,
    filter_x: np.ndarray,
    filter_y: np.ndarray,
) -> np.ndarray:
    """Convenience function to apply filters along the x and y axes

    Args:
        data (np.ndarray): data array, expected to be 2D but could be more
            (filters will only be applied to the last two dimensions)
        filter_x (np.ndarray): padded fourier-space filter in the x direction
            which corresponds to the second-to-last axis
        filter_y (np.ndarray): padded fourier-space filter in the y direction
            which corresponds to the last axis

    Returns:
        result (np.ndarray): data after the filters have been applied
    """
    post_x = apply_filter_fourier_1d(data, filter_x, axis=-1)
    result = apply_filter_fourier_1d(post_x, filter_y, axis=-2)

    return result