import numpy as np
import torch
from scipy.special import hankel1
from scipy.integrate import quad


def generate_gaussian_potential(
    eta_domain: float, n: int, contrast: float
) -> np.ndarray:
    """Generates a Gaussian, centered at 0, with sigma  val 0.1, scaled to have maximum
    height <contrast>.

    Args:
        eta_domain (float): Half of the side length of the scattering domain. The
                    scattering domain will be [-eta_domain, eta_domain]^2
        n (int): Number of pixels along each side of the scattering domain.
        contrast (float): The contrast of the scattering objects.

    Returns:
        np.ndarray: Shape (n, n)
    """
    x = np.linspace(-eta_domain, eta_domain, n)
    y = np.linspace(-eta_domain, eta_domain, n)
    [X, Y] = np.meshgrid(x, y)
    coords = np.stack((X, Y), axis=-1)
    nrms = np.linalg.norm(coords, axis=-1)
    return contrast * _gaussian(nrms, 0.1)


def greensfunction2(
    gpt: np.ndarray, k: float, dx: float, diag_correction: float = None
) -> np.ndarray:
    """Computes the interior Green's function for the free-space Helmholtz
    equation, evaluated at the points gpts

    out[i, j] = G(||gpt[i] - gpt[j] ||)

    Args:
        gpt (np.ndarray): Has shape (n_pixels ** 2, 2)
        k (float): Frequency
        dx (float): cell spacing
        diag_correction (float): The correction for the singularity of the
        Green's function when evaluated at 0.

    Returns:
        np.ndarray: has shape (n_pixels ** 2, n_pixels ** 2)
    """
    G = np.zeros((gpt.shape[0], gpt.shape[0]), dtype=complex)
    for i in range(gpt.shape[0]):
        kR = k * np.sqrt(np.sum((gpt - gpt[i, :]) ** 2, axis=1))
        tmp = hankel1(0, kR)
        G[i, :] = -1j / 4 * tmp

    if diag_correction is not None:
        np.fill_diagonal(G, diag_correction)
    else:
        np.fill_diagonal(G, 0)
    G = G * (dx**2)
    return G


def greensfunction3(
    gpt: np.ndarray, k: float, dx: float, diag_correction: float = None
) -> np.ndarray:
    """This computes a Greens function and stores the results in an array.
    Given input points gpt, the output is

    out[i, j] = G(|| gpt[i, j] ||)

    This function assumes || gpt[0, 0] || = 0, and [0, 0] is the only place in
    the gpt array where this is satisfied. If this assumption is not met, the
    diag correction will be applied incorrectly.


    Args:
        gpt (np.ndarray): Has shape (n_pixels, n_pixels, 2)
        k (float): Frequency
        dx (float): cell spacing
        diag_correction (float): The correction for the singularity of the
        Green's function when evaluated at 0.

    Returns:
        np.ndarray: has shape (n_pixels, n_pixels). Complex-valued array.
    """

    nrms = np.linalg.norm(gpt, axis=2)
    tmp = hankel1(0, k * nrms)
    out = -1j / 4 * tmp

    if diag_correction is not None:
        out[0, 0] = diag_correction
    else:
        out[0, 0] = 0.0

    out = out * (dx**2)
    return out


def _extend_1d_grid(n_small: int, dx: float) -> np.ndarray:
    """Helper function for get_extended_grid."""
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


def getGscat2circ(
    gpts: np.ndarray, gpts_circ: np.ndarray, k: float, dx: float
) -> np.ndarray:
    """Computes the outgoing greens function mapping from the scattering domain
    to the reciver points.

    Args:
        gpts (np.ndarray): Scattering domain quadrature points. Has shape
                            (n_quad, 2)
        gpts_circ (np.ndarray): receiver points. Has shape (n_rec, 2)
        k (float): Frequency

    Returns:
        np.ndarray: Has shape (n_quad * n_rec, n_quad * n_rec) and is dense
    """
    G = np.zeros((gpts_circ.shape[0], gpts.shape[0]), dtype=complex)
    for i in range(gpts_circ.shape[0]):
        kR = k * np.sqrt(np.sum((gpts_circ[i, :] - gpts) ** 2, axis=1))
        tmp = hankel1(0, kR)
        tmp = -1j / 4 * tmp
        G[i, :] = tmp
    G = G * (dx**2)
    return G


def _gaussian(rad: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-1 * np.square(rad) / (sigma**2))


def find_diag_correction(h: float, k: float) -> float:
    """
    This function computes the correction for the singularity of
    the Helmholtz operator's Green's function at the origin.

    Here's what it does:
    1. make a 1D grid with bounds [-5, 5] and spacing h
    2. make f(x) = narrow Gaussian centered at 0
    3. make g(x) = Green's function
    4. Compute int f(x) g(x) on grid defined above via punctured trapezoid rule
      (zero out g(x) at singularity)
    5. Compute int f(x) g(x) on domain defined above with some adaptive integration
      (fron scipy)
    6. Return diff / (h ** 2)
    """
    MAX = 5

    SIGMA_VAL = 0.8

    # Set up grid
    n_grid_points = int(2 * MAX / h)
    x = np.arange(n_grid_points) * h
    # x = np.linspace(0, 2 * MAX, n_grid_points, endpoint=False)
    zero_idx = n_grid_points // 2
    x = x - x[zero_idx]
    assert 0 in x
    assert np.allclose(h, x[1] - x[0]), f"{h}, {x[0]}, {x[1]}, {x[1] - x[0]}"

    # Set up 2D mesh
    X, Y = np.meshgrid(x, x)
    mesh = np.stack((X, Y), axis=-1)

    # Set up the Gaussian evals. I need to zero off some of the Gaussian in the
    # corner of the computational domain because I am comparing against
    # integration by polar coordinates.
    nrms = np.linalg.norm(mesh, axis=2)
    bool_arr = nrms < MAX
    bin_arr = bool_arr.astype(np.float32)
    f_evals_square = _gaussian(nrms, SIGMA_VAL) * bin_arr
    f_evals_vec = f_evals_square.flatten()

    # Set up a G array
    extended_domain_grid = get_extended_grid(n_grid_points, h)
    G_int = greensfunction3(extended_domain_grid, k, h)
    G_fft = np.fft.fft2(G_int)
    out_vec = _G_apply_numpy(f_evals_vec, G_fft, n_grid_points)
    out_mat = out_vec.reshape(f_evals_square.shape)

    punctured_trap_val = out_mat[zero_idx, zero_idx]

    # We want to integrate f * g in polar coordinates. So we do:
    def _int_eval_func(r: float) -> float:
        abs_r = np.abs(r)
        f = _gaussian(abs_r, SIGMA_VAL)
        g = -1j / 4 * hankel1(0, k * abs_r)
        return (f * g) * abs_r * np.pi

    adaptive_int_val, _ = quad(
        _int_eval_func, -MAX, MAX, complex_func=True, points=(0,), limit=100
    )

    diff = (adaptive_int_val - punctured_trap_val) / ((h**2))
    return diff


def _G_apply_numpy(x: np.ndarray, G_fft: np.ndarray, N: int) -> np.ndarray:
    """Same as HelmholtzSolverAccelerated._G_apply() routine, written in numpy.
    Sometimes helpful for debugging in envs where GPU is not available."""
    x_shape = x.shape
    x_square = x.reshape(N, N)

    x_pad = _zero_pad_numpy(x_square, G_fft.shape[0])

    x_fft = np.fft.fft2(x_pad)

    out_fft = np.fft.ifft2(G_fft * x_fft)

    out = out_fft[:N, :N]

    o = out.reshape(x_shape)
    # logging.debug("_fast_G_apply: output shape: %s", o.shape)

    return o


def _zero_pad_numpy(v: np.ndarray, n: int) -> np.ndarray:
    """Same as HelmholtzSolverAccelerated._zero_pad() routine, written in numpy.
    Sometimes helpful for debugging in envs where GPU is not available."""
    o = np.zeros((n, n), dtype=v.dtype)
    o[: v.shape[0], : v.shape[1]] = v

    return o
