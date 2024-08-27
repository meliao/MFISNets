import numpy as np
from typing import Tuple
import torch


def get_freq_grid(real_grid: np.ndarray) -> np.ndarray:
    """Given a 2D grid in real space, returns a 2D grid in frequency space
    corresponding to the grid points that the DFT would output.

    Args:
        real_grid (np.ndarray): Represents a grid of N x N quadrature points. Has shape (N, N, 2)

    Returns:
        np.ndarray: Has shape (N, N, 2). Represents the quadrature points in Fourier space.
    """
    n = real_grid.shape[0]

    dx = real_grid[0, 1, 0] - real_grid[0, 0, 0]
    big_grid = get_extended_grid(n, dx)
    big_n = big_grid.shape[0]
    freqs_1d = torch.fft.fftshift(torch.fft.fftfreq(big_n, d=dx)).numpy()
    freqs_x, freqs_y = np.meshgrid(freqs_1d, freqs_1d)
    freqs_grid = np.stack((freqs_x, freqs_y), axis=-1)
    center_x = big_n // 2 - n // 2
    freqs_grid_out = freqs_grid[center_x : center_x + n, center_x : center_x + n]
    return freqs_grid_out


def fourier_transform_2d(
    signal: np.ndarray, grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pads the signal to grid in 3x each dimension, then takes the FFT,
    then returns the un-padded stuff. Also returns the corresponding
    frequency grid.

    Args:
        signal (np.ndarray): Input signal in real space, unpadded
        grid (np.ndarray): The (N, N, 2) grid corresponding to the signal

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of the signal in frequency space
            and the frequency grid.
    """
    n = grid.shape[0]

    signal_torch = torch.from_numpy(signal)
    dx = grid[0, 1, 0] - grid[0, 0, 0]
    assert dx != 0
    big_grid = get_extended_grid(n, dx)

    big_n = big_grid.shape[0]

    signal_padded = _zero_pad(signal_torch, big_n)
    signal_fft = torch.fft.fft2(signal_padded, norm="ortho")

    # Get the sampling frequencies
    freqs_1d = torch.fft.fftfreq(big_n, d=dx)

    # Do a shift so the small frequencies are in the center of the array
    signal_fft = torch.fft.fftshift(signal_fft)
    freqs_1d = torch.fft.fftshift(freqs_1d).numpy()

    # Make a grid of the frequencies
    freqs_x, freqs_y = np.meshgrid(freqs_1d, freqs_1d)
    freqs_grid = np.stack((freqs_x, freqs_y), axis=-1)

    # Only look at the low frequencies:
    center_x = big_n // 2 - n // 2
    out = signal_fft[center_x : center_x + n, center_x : center_x + n]
    freqs_grid_out = freqs_grid[center_x : center_x + n, center_x : center_x + n]

    return out.numpy(), freqs_grid_out


# def random_signal_from_fourier_mask_2d(
#     mask: np.ndarray, grid: np.ndarray
# ) -> np.ndarray:
#     """_summary_

#     Args:
#         signal (np.ndarray): _description_
#         grid (np.ndarray): _description_

#     Returns:
#         np.ndarray: _description_
#     """
#     mask_torch = torch.from_numpy(mask)
#     n = grid.shape[0]
#     dx = grid[0, 1, 0] - grid[0, 0, 0]
#     assert dx != 0
#     big_grid = get_extended_grid(n, dx)
#     big_n = big_grid.shape[0]

#     mask_padded = _zero_pad_in_center(mask_torch, big_n)
#     random_fourier_coeffs = torch.randn(mask_padded.shape) + 1j * torch.randn(
#         mask_padded.shape
#     )

#     prod = mask_padded * random_fourier_coeffs

#     prod = torch.fft.ifftshift(prod)

#     out_padded = torch.fft.ifft2(prod)
#     return out_padded[:n, :n].real.numpy()


def inverse_fourier_transform_2d(
    signal_fourier: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Performs the reverse direction of the fourier_transform_2d function
    Args:
        signal_fourier (np.ndarray): Input signal in Fourier space, unpadded
        grid (np.ndarray): The (N, N, 2) frequency grid corresponding to the signal

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of the signal in real space
            and the real space grid.
    """
    n = grid.shape[0]
    dx = grid[0, 1, 0] - grid[0, 0, 0]
    assert dx != 0
    big_grid = get_extended_grid(n, dx)
    big_n = big_grid.shape[0]

    signal_padded = _zero_pad_in_center(
        torch.from_numpy(signal_fourier), big_grid.shape[0]
    )
    signal_padded = torch.fft.ifftshift(signal_padded)

    signal_pixel_space = torch.fft.ifft2(signal_padded, norm="ortho")
    center_x = big_n // 2 - n // 2
    # out = signal_pixel_space[center_x : center_x + n, center_x : center_x + n]
    out = signal_pixel_space[:n, :n]

    return out.real.numpy()


def _zero_pad(v: torch.Tensor, n: int) -> torch.Tensor:
    """v has shape (N_pixels, N_pixels) output has shape (n, n)"""
    o = torch.zeros((n, n), dtype=v.dtype)
    o[: v.shape[0], : v.shape[1]] = v
    return o


def _zero_pad_in_center(v: torch.Tensor, big_n: int) -> torch.Tensor:
    """v has shape (n, n) and output has shape (big_n, big_n)"""
    o = torch.zeros((big_n, big_n), dtype=v.dtype)
    n = v.shape[0]
    center_x = big_n // 2 - n // 2
    o[center_x : center_x + n, center_x : center_x + n] = v
    return o


def _extend_1d_grid(n_small: int, dx: float) -> np.ndarray:
    """Returns a padded copy of a 1d grid given the spacing dx
    and number of grid points n_small in the center.

    Helper function for the 2D grid extension function below
    """
    target_n_points = 3 * n_small
    two_exp = int(round(np.log2(target_n_points)))
    n_grid = 2**two_exp

    half_n = n_grid // 2
    lim = half_n * dx
    grid = dx * np.arange(n_grid)
    grid = grid - grid[half_n]

    # grid = np.linspace(-lim, lim, n_grid, endpoint=False)
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
