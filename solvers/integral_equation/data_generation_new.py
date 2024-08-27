import numpy as np

from solvers.integral_equation.random_shape_generation import (
    _random_square,
    _random_ellipse,
    _random_triangle,
)

from src.utils.scale_separation_utils import (
    fourier_transform_2d,
    inverse_fourier_transform_2d,
    get_freq_grid,
)
from src.data.lowpass_filter import apply_lpf, apply_filter_fourier_2d


def shapes_and_non_constant_background(
    n_shapes: int,
    height: float,
    domain_max: float,
    n_points: int,
    background_max_freq: float,
    background_max_radius: float,
    no_intersection_bool: bool = False,
    lpf_obj: np.ndarray | None = None,
    use_tuple_lpf: bool = False,
    # use lpf_obj as a tuple of lpfs along one dimension
    # as in the output of prep_lpf_from_wavenum
) -> np.ndarray:
    """This function is used to draw samples from our distribution of
    scattering potentials, which have piecewise-constant geometric
    shapes and non-constant low-frequency background elements.

    Args:
        n_shapes (int): How many shapes should be generated in the image
        height (float): Max value of the image. Also called the contrast.
        domain_max (float): Half side length of the spatial domain, which is a square centered at the origin.
        n_points (int): How many pixels along one side of the image.
        background_max_freq (float): Frequency parameter for the background
        background_max_radius (float): How far the background should extend from the origin
        no_intersection_bool (bool, optional): If True, random shape generation is constrained so the shapes are non-overlapping. Defaults to False.
        lpf_obj (np.ndarray | None, optional): A lowpass filter used to slightly smooth the scattering objects after generation. Defaults to None.
        use_tuple_lpf (bool, optional): If the lpf_obj is a Tuple representing the smoothing in x and y directions, set this to True. Defaults to False.

    Returns:
        np.ndarray: _description_
    """
    x = np.linspace(-domain_max, domain_max, n_points, endpoint=False)
    X, Y = np.meshgrid(x, x)
    xy_grid = np.stack((X, Y), axis=-1)
    nrm_grid = np.linalg.norm(xy_grid, axis=-1)

    q = _random_shapes(n_shapes, height, X, Y)
    if no_intersection_bool:
        while q.max() > height:
            q = _random_shapes(n_shapes, height, X, Y)

    q_mask = q > 0

    background = _non_constant_background(
        height,
        background_max_freq,
        X,
        Y,
    )
    rad_mask = nrm_grid < background_max_radius
    background = background * rad_mask

    background[q_mask] = height

    if lpf_obj is not None:
        if use_tuple_lpf:
            background = apply_filter_fourier_2d(background, lpf_obj[0], lpf_obj[1])
        else:
            background = apply_lpf(background[np.newaxis, :], lpf_obj).squeeze(0)
    return background


def _random_shapes(
    N_SHAPES: int, HEIGHT: float, x1: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    """Generates random shapes

    Args:
        N_SHAPES (int): Number of shapes
        HEIGHT (float): Contrast level
        x1 (np.ndarray): X values of a meshgrid represnting the scattering domain. Has shape (n_x, n_x)
        x2 (np.ndarray): Y values of a meshgrid represnting the scattering domain. Has shape (n_x, n_x)

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: Has shape (n_x, n_x).
    """
    aux = np.zeros_like(x1)
    for gi in range(N_SHAPES):
        # Draw shape type
        shape_type_int = np.random.randint(1, 4)
        if shape_type_int == 1:
            # Draw random square params

            theta = 2 * np.pi * np.random.uniform()

            # side_len uniform [0.1, 0.15]
            side_len = 0.1 + 0.05 * np.random.uniform()

            # pick center so that square will always be inside unit circle.
            center_x = 0.3 * np.random.uniform() - 0.15
            center_y = 0.3 * np.random.uniform() - 0.15

            # Update aux
            aux += _random_square(center_x, center_y, side_len, theta, HEIGHT, x1, x2)

        elif shape_type_int == 2:
            theta = 2 * np.pi * np.random.uniform()

            # side_len uniform [0.1, 0.15]
            side_len = 0.1 + 0.05 * np.random.uniform()

            # pick center so that triangle will always be inside the unit circle.
            center_x = 0.3 * np.random.uniform() - 0.15
            center_y = 0.3 * np.random.uniform() - 0.15

            # Update aux
            aux += _random_triangle(center_x, center_y, side_len, theta, HEIGHT, x1, x2)

        elif shape_type_int == 3:
            theta = 2 * np.pi * np.random.uniform()

            # side_len_1 uniform [0.1, 0.15]
            side_len_1 = 0.1 + 0.05 * np.random.uniform()
            # side_len_2 uniform [0.05, 0.1]
            side_len_2 = 0.05 + 0.05 * np.random.uniform()

            center_x = 0.2 * np.random.uniform() - 0.1
            center_y = 0.2 * np.random.uniform() - 0.1

            aux += _random_ellipse(
                center_x, center_y, side_len_1, side_len_2, theta, HEIGHT, x1, x2
            )

        else:
            raise ValueError
    return aux


def _gaussian(nrms: np.ndarray, param) -> np.ndarray:
    return np.exp(-1 * (nrms**2) / (2 * param**2))


def _non_constant_background(
    height: float,
    max_freq: float,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """Generates a random non-constant background by drawing random Fourier modes.

    Args:
        height (float): Max contrast
        max_freq (float): Maximum frequency content
        xx (np.ndarray): X samples of meshgrid represnting scattering domain. Has shape (n_x, n_x)
        yy (np.ndarray): Y samples of meshgrid represnting scattering domain. Has shape (n_x, n_x)

    Returns:
        np.ndarray: Has shape (n_x, n_x)
    """
    xy_grid = np.stack((xx, yy), axis=-1)
    freq_grid = get_freq_grid(xy_grid)

    freq_nrms = np.linalg.norm(freq_grid, axis=-1)
    # xy_nrms = np.linalg.norm(xy_grid, axis=-1) # unused at the moment

    # use max_freq as the HWHM level
    sig_in_freq_space = max_freq / np.sqrt(2 * np.log(2))

    glpf_mask = np.exp(-0.5 * (freq_nrms / sig_in_freq_space) ** 2)

    random_fourier_coeffs = (
        1
        / np.sqrt(2)
        * (
            np.random.normal(size=freq_nrms.shape)
            + 1j * np.random.normal(size=freq_nrms.shape)
        )
    )

    random_fourier_coeffs = random_fourier_coeffs * glpf_mask

    background = inverse_fourier_transform_2d(random_fourier_coeffs, xy_grid)

    # minimum value = 0.
    background = background - background.min()

    # maximum value = height
    background_max = background.max()
    background = background * height / background_max

    # g = _gaussian(xy_nrms, gaussian_param)
    return background
