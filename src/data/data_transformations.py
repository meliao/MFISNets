import numpy as np
import torch
import scipy.sparse

from typing import Tuple, Callable


# these are the constants for the integral equation solver with omega = 16 * 2 * pi
# We computed these values by doing the following:
# 1. Generating a Gaussian scattering potential with very small perturbation,
#    max contrast of 1e-04, and a width of 0.1. This was to ensure we were in
#    the regime where the linear approximation to the forward model would be accurate.
# 2. Evaluate the forward model at a single frequency of omega = 16 * 2 * pi by
#    using the integral equation solver code in solvers.integral_equation directory,
#    and then trasnform the output to (m, h) coordinates.
# 3. Evaluate the forward model at a single frequency of omega = 16 * 2 * pi by
#    using the linearized forward model derived by Fan and Ying 2022, which outpus in
#    (m, h) coordinates.
# 4. Compare the two results and find a phase and amplitude shift that minimizes the
#    difference between the two results.
CONST_RHO_PRIME = 25569.7
CONST_THETA_PRIME = 0.79025


def get_scale_factor(rho_prime: float, theta_prime: float) -> np.complex64:
    a = np.sqrt(rho_prime)
    b = np.exp(-1 * 1j * theta_prime)
    return a * b


### Functions for precomputing the interpolation stage ###


def prep_conv_interp_1d(
    points: np.ndarray,
    xi: np.ndarray,
    bc_mode: str = None,
    # periodic: bool = False,
    a_neg_half: bool = True,
) -> scipy.sparse.csr_array:
    """Prepares a sparse array to apply convolution for cubic interpolation
    Operates in a single dimension and can be applied to each dimension independently
    to work with higher-dimension data

    Assumes that the entries in `points` are sorted and evenly spaced
    Args:
        points (ndarray): original data grid points
        xi (ndarray): array of points to be sampled as a (m,)-shaped array
        bc_mode (string): how to handle the boundary conditions
            options:
                "periodic": wrap values around
                "extend": extrapolate the missing out-of-boundary values
                    using a rule for cartesian points: f(-1) = 3*f(0) - 3*f(1) + f(2)
                    (see R. Keys 1981 paper below)
                "zero": sets values outside the boundary to zero
        a_neg_half (bool): whether to use a=-1/2 for the convolution filter (otherwise use a=-3/4)
    Returns:
        conv_filter (m by n): sparse linear filter to perform
            cubic interpolation (on padded data)
            Apply to data values with `apply_interp_{1,2}d`
            Note: padding should not be needed except for the inside edge of a polar grid

    For the choice of convolution filter for cubic convolution
    See https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    and R. Keys (1981). "Cubic convolution interpolation for digital image processing".
        IEEE Transactions on Acoustics, Speech, and Signal Processing.
        29 (6): 1153–1160. Bibcode:1981ITASS..29.1153K. CiteSeerX 10.1.1.320.776.
        doi:10.1109/TASSP.1981.1163711 .
    """
    bc_mode = bc_mode.lower() if bc_mode is not None else "zero"
    # Helper variables
    periodic_mode = bc_mode == "periodic"
    extend_mode = bc_mode == "extend"

    # Set up the 1D filter to be used
    if a_neg_half:
        # with a=-1/2, standard choice (seems to be the Catmull-Rom filter?)
        cubic_conv_matrix = 0.5 * np.array(
            [[0, 2, 0, 0], [-1, 0, 1, 0], [2, -5, 4, -1], [-1, 3, -3, 1]]
        )
    else:
        # with a=-3/4, computed with sympy
        # Sometimes gives lower error but has weaker theoretical properties...
        cubic_conv_matrix = 0.25 * np.array(
            [[0, 4, 0, 0], [-3, 0, 3, 0], [6, -9, 6, -3], [-3, 5, -5, 3]]
        )
    n = points.shape[0]
    min_pt = points[0]
    itvl = (points[-1] - points[0]) / (
        n - 1
    )  # interval between regularly sampled points
    m = xi.shape[0]

    # Faster to build in LIL form then convert later to CSR
    interp_op = scipy.sparse.lil_array((m, n))

    for i, x in enumerate(xi):
        j_float, x_offset = np.divmod(x - min_pt, itvl)
        j = int(j_float)
        pos_rel = x_offset / itvl
        monomial_vec = np.array([1, pos_rel, pos_rel**2, pos_rel**3])
        filter_local = monomial_vec @ cubic_conv_matrix

        if j >= 1 and j <= n - 3:
            interp_op[i, j - 1 : j + 3] = filter_local
        elif periodic_mode:
            # Wrap around the input values
            j_idcs = np.arange(j - 1, j + 3) % n
            interp_op[i, j_idcs] = filter_local
        elif extend_mode:
            # Assumes zero value beyond the extra single-cell padding
            # Extrapolation rule was linear anyway so fold down the extrapolation
            # into a reduced-length filter
            if j < 1 and (j + 3) >= 0:
                filter_folded = filter_local[1:] + filter_local[0] * np.array(
                    [3, -3, 1]
                )
                interp_op[i, : j + 3] = filter_folded[: j + 3]
            elif j < n and (j + 3) >= n:
                filter_folded = filter_local[:-1] + filter_local[-1] * np.array(
                    [1, -3, 3]
                )
                interp_op[i, j - 1 :] = filter_folded[: n - j + 1]
        else:  # bc_mode == "zero" case
            if j < 1 and (j + 3) >= 0:
                interp_op[i, : j + 3] = filter_local[: j + 3]
            elif j < n and (j + 3) >= n:
                interp_op[i, j - 1 :] = filter_local[: n - j + 1]

    # Convert for slightly faster application
    interp_op = scipy.sparse.csr_array(interp_op)
    return interp_op


def prep_conv_interp_2d(
    points_x: np.ndarray,
    points_y: np.ndarray,
    xi: np.ndarray,
    bc_modes: None | str | Tuple = None,
    a_neg_half: bool = True,
) -> Tuple[scipy.sparse.csr_array, scipy.sparse.csr_array]:
    """Wrapper function for prep_conv_interp_1d using the same boundary condition modes
    Args:
        points_x (ndarray): original data grid point along the x-axis
        points_y (ndarray): original data grid point along the y-axis
        xi (ndarray): array of points to be sampled as a (m, 2)-shaped array
        bc_modes (string or tuple of strings): how to handle the boundary conditions
            Note: Can use different boundary conditions for each dimension
            by passing bc_modes as a tuple of strings
            Options:
                "periodic": wrap values around
                "extend": extrapolate the missing out-of-boundary values
                    using a rule for cartesian points: f(-1) = 3*f(0) - 3*f(1) + f(2)
                    (see R. Keys 1981 paper below)
                "zero": sets values outside the boundary to zero
        a_neg_half (bool): whether to use a=-1/2 for the convolution filter (otherwise uses a=-3/4)
    Returns:
        interp_op_x, interp_op_y (scipy csr_arrays): sparse arrays as linear operators to perform convolution
            Each operator can be applied separately onto each side of the data matrix
            (also provided below are convenience functions `apply_interp_1d` and `apply_interp_2d`)
    """
    should_split = hasattr(bc_modes, "__len__") and len(bc_modes) == 2
    bc_mode_x = bc_modes[0] if should_split else bc_modes
    bc_mode_y = bc_modes[1] if should_split else bc_modes
    interp_op_x = prep_conv_interp_1d(
        points_x, xi[:, 0], bc_mode=bc_mode_x, a_neg_half=a_neg_half
    )
    interp_op_y = prep_conv_interp_1d(
        points_y, xi[:, 1], bc_mode=bc_mode_y, a_neg_half=a_neg_half
    )
    return interp_op_x, interp_op_y


def prep_polar_padder(
    polar_grid_r: np.ndarray, ntheta: int, dim: int = 0, with_torch: bool = False
) -> Tuple[np.ndarray, Callable]:
    """Prepare a padding function for the origin-side of a polar grid
    Idea: prepare a function to perform the padding for a given ntheta
    Note: it appears that if the function is relatively smooth, this may not be necessary...

    Args:
        polar_grid_r (ndarray): array of the radii for the polar grid points
        ntheta (int): number of angles used in the polar grid (actual locations not needed)
        dim (int): the dimension that behaves as the radius dimension, by default set to 0
        with_torch (bool): use torch operations rather than numpy

    Returns:
        padded_polar_grid_r: like polar_grid_r but with an extra gridpoint appended
            (for easy filter construction/application)
        polar_padder: function that performs padding operation to find the right value across the origin
    """
    nr = polar_grid_r.shape[0]
    r_itvl = polar_grid_r[1] - polar_grid_r[0]

    # First, compute the padded grid points
    padded_polar_grid_r = np.array([polar_grid_r[0] - r_itvl, *polar_grid_r])

    # Next, prepare a function that can fill in the missing values
    if dim == 0:
        dim_stack = torch.row_stack if with_torch else np.row_stack
        inner_ring_slice = np.s_[1, :]
    else:
        dim_stack = torch.column_stack if with_torch else np.column_stack
        inner_ring_slice = np.s_[:, 1]
    roll_fn = torch.roll if with_torch else np.roll

    if ntheta % 2 == 0:

        def polar_padder(polar_data):
            """Pad data corresponding to a polar grid (even ntheta grid version)
            Since there is an even number of data points, we can just cycle the points
            in the inner ring of the polar grid to find the values across the origin
            """
            nonlocal inner_ring_slice, dim_stack, roll_fn
            # print(inner_ring_slice, dim_stack)
            padding_data = roll_fn(polar_data[inner_ring_slice], ntheta // 2)
            padded_polar_data = dim_stack([padding_data, polar_data])
            return padded_polar_data

    else:
        # Construct a periodic interpolator.. probably would break torch autodiff though
        data_grid_angles = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        interp_grid_angles = np.mod(
            data_grid_angles - np.pi, 2 * np.pi
        )  # get the opposite angle
        padding_interp_op = prep_conv_interp_1d(
            data_grid_angles, interp_grid_angles, bc_mode="periodic", a_neg_half=True
        )
        padding_interp_op = torch.tensor(
            padding_interp_op.todense(), dtype=torch.float32, requires_grad=False
        )

        def polar_padder(polar_data):
            """Pad data corresponding to a polar grid (odd ntheta grid version)
            Because there is an odd number of data points, we perform an interpolation operation
            """
            nonlocal inner_ring_slice, dim_stack, padding_interp_op

            # print("polar_padder: padding_interp_op type: ", type(padding_interp_op))
            # print("polar_padder: inner_ring_slice type: ", type(inner_ring_slice))
            # print("polar_padder: polar_data type: ", type(polar_data))
            # Apply the periodic interpolator to get padding_data
            padding_data = padding_interp_op @ polar_data[inner_ring_slice]
            padded_polar_data = dim_stack([padding_data, polar_data])
            return padded_polar_data

    return padded_polar_grid_r, polar_padder


def apply_interp_1d(
    interp_op: scipy.sparse.csr_array, data_vals: np.ndarray
) -> np.ndarray:
    """Applies the x/y convolutional filter operators onto the padded values
        ~ diag(conv_x (vals) conv_y^T)
    interp_op is   m x nx
    data_vals is nx x {1, ny}
    result is m (or m x ny) interpolated values
    """
    res = interp_op @ data_vals  # .sum(1) # (m, ny+2) to (m,)
    return res


def apply_interp_2d(
    interp_op_x: scipy.sparse.csr_array,
    interp_op_y: scipy.sparse.csr_array,
    data_vals: np.ndarray,
) -> np.ndarray:
    """Applies the x/y convolutional filters onto the padded values
        ~ diag(conv_x (vals) conv_y^T)
    Args:
        interp_op_x (sparse csr array): x-dim interpolation operator with shape (m, nx)
        interp_op_y (sparse csr array): y-dim interpolation operator with shape (m, ny)
        data_vals (ndarray): data values arranged in a grid (nx, ny)
    Returns:
        result (ndarray) is the interpolated values in a (m,) array
    """
    post_x = interp_op_x @ data_vals  # m x (ny+2)
    res = (post_x * interp_op_y).sum(1)
    return res


def prep_rs_to_mh_interp(
    grid_r: np.ndarray, grid_s: np.ndarray, n_m: int, n_h: int, a_neg_half: bool = True
) -> Tuple[scipy.sparse.csr_array, scipy.sparse.csr_array]:
    """Prepares interpolation operators from (r, s) to (m, h) for the wave field measurements
    Args:
        grid_r (ndarray): grid points for the r angles
        grid_s (ndarray): grid points for the s angles
        n_m (int): number of m points in the output grid
        n_h (int): number of h points in the output grid
        a_neg_half (bool): interpolation mode for "a" parameter to be passed to prep_conv_interp_2d
    Returns:
        rs_to_mh_r (sparse csr_array): interpolation operator for the r-dimension
        rs_to_mh_s (sparse csr_array): interpolation operator for the s-dimension
    """
    grid_m = np.linspace(0, 2 * np.pi, n_m, endpoint=False)
    grid_h = np.linspace(
        -np.pi / 2, np.pi / 2, n_h, endpoint=False
    )  # h has a reduced range

    grid_mh = np.array(np.meshgrid(grid_m, grid_h)).T.reshape(n_m * n_h, 2)
    # grid_mh = np.array(np.meshgrid(grid_m, grid_h)).transpose(1,2,0).reshape(n_m*n_h, 2)
    grid_mh_in_rs_coords = np.array(
        [
            np.mod(grid_mh[:, 0] + grid_mh[:, 1], 2 * np.pi),  # r
            np.mod(grid_mh[:, 0] - grid_mh[:, 1], 2 * np.pi),  # s
        ]
    ).T

    conv_rs_to_m, conv_rs_to_h = prep_conv_interp_2d(
        grid_r,
        grid_s,
        grid_mh_in_rs_coords,
        bc_modes=(
            "periodic",
            "periodic",
        ),  # both dimensions are angles and therefore periodic
        a_neg_half=a_neg_half,
    )

    return conv_rs_to_m, conv_rs_to_h


def apply_interp_2d_batched(
    interp_op_x: torch.sparse_csr_tensor,
    interp_op_y: torch.sparse_coo_tensor,
    data_vals: torch.tensor,
) -> torch.tensor:
    """apply_interp_2d for batched inputs.. so data_vals takes shape (nb, nx, ny)
    Note: intended for torch tensors

    Applies the x/y convolutional filters onto the padded values
        ~ diag(conv_x (vals) conv_y^T)
    Args:
        interp_op_x (torch sparse csr tensor): x-dim interpolation operator with shape (m, nx)
        interp_op_y (torch sparse coo tensor): y-dim interpolation operator with shape (m, ny)
            Note: it's a headache working with csr from this side with torch
        data_vals (torch tensor): data values arranged in a grid (nx, ny)
    Returns:
        result (torch tensor) is the interpolated values in a (nb, m) array
    """
    # print(f"data vals shape: {data_vals.shape}")
    # assert data_vals.ndim==3
    if getattr(data_vals, "ndim", None) != 3:
        # Try calling the base version instead
        return apply_interp_2d(interp_op_x, interp_op_y, data_vals)
    # if interp_op_y.layout != torch.sparse_coo:
    #     print(f"(apply_interp_2d_batched) BEWARE -- converting interp_op_y to a COO array every time")
    #     interp_op_y = interp_op_y.to_sparse_coo()

    nb, nx, ny = data_vals.shape
    m = interp_op_x.shape[0]
    post_x = interp_op_x @ data_vals
    res = (post_x * interp_op_y[np.newaxis, ...]).sum(-1).to_dense()
    return res


def polar_pad_and_apply(
    polar_padder: Callable,
    interp_op_x: scipy.sparse.csr_array,
    interp_op_y: scipy.sparse.csr_array,
    data_vals: np.ndarray,
    batched: bool = False,
) -> np.ndarray:
    """Convenience function to pad polar grids then apply the interpolation operators
    For simplicity, assume that data_vals is laid out as (n_r, n_theta)

    Args:
        polar_padder (function): padding helper function from prep_polar_padder
        interp_op_x (sparse csr array): polar-to-x operator
        interp_op_y (sparse csr array): polar-to-y operator
        data_vals (np npdarray): (n_r, n_theta) data array
        batched (bool): whether to treat data_vals as (n_batch, n_r, n_theta) instead

    Returns:
        res_vals (np ndarray): flattened array of values at the interpolated points -- needs to be reshaped
    """

    if batched:
        assert data_vals.ndim == 3
        padded_vals = torch.stack(
            [polar_padder(data_vals[i]) for i in range(data_vals.shape[0])], axis=0
        )
        res_vals = apply_interp_2d_batched(interp_op_x, interp_op_y, padded_vals)
    else:
        padded_vals = polar_padder(data_vals)
        res_vals = apply_interp_2d(interp_op_x, interp_op_y, padded_vals)
    return res_vals


def polar_to_euclidean(theta: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Creates an array of Euclidean coordinates for the specified theta

    Args:
        theta (np.ndarray): Has shape N_theta
        rho (np.ndarray): Has shape N_rho

    Returns:
        np.ndarray: Has shape (N_theta * N_rho, 2)
    """
    theta_mesh, rho_mesh = np.meshgrid(theta, rho, indexing="ij")
    theta_mesh = theta_mesh.flatten()
    rho_mesh = rho_mesh.flatten()
    x = rho_mesh * np.cos(theta_mesh)
    y = rho_mesh * np.sin(theta_mesh)
    return np.vstack((y, x)).T
