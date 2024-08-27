import numpy as np
import pytest
import torch
from solvers.integral_equation.Helmholtz_solver_utils import (
    greensfunction2,
    greensfunction3,
    getGscat2circ,
    find_diag_correction,
)
from solvers.integral_equation.HelmholtzSolver import (
    HelmholtzSolverAccelerated,
    setup_accelerated_solver,
)

N_PIXELS = 50
SPATIAL_DOMAIN_MAX = 0.5
WAVENUMBER = 16
RECEIVER_RADIUS = 100
SOLVER_OBJ = setup_accelerated_solver(
    N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
)


def check_arrays_close(
    a: np.ndarray,
    b: np.ndarray,
    a_name: str = "a",
    b_name: str = "b",
    msg: str = "",
    atol: float = 1e-8,
    rtol: float = 1e-05,
) -> None:
    s = _evaluate_arrays_close(a, b, msg, atol, rtol)
    allclose_bool = np.allclose(a, b, atol=atol, rtol=rtol)
    assert allclose_bool, s


def _evaluate_arrays_close(
    a: np.ndarray,
    b: np.ndarray,
    msg: str = "",
    atol: float = 1e-08,
    rtol: float = 1e-05,
) -> str:
    assert a.size == b.size, f"Sizes don't match: {a.size} vs {b.size}"

    max_diff = np.max(np.abs(a - b))
    samp_n = 5

    # Compute relative difference
    x = a.flatten()
    y = b.flatten()
    difference_count = np.logical_not(np.isclose(x, y, atol=atol, rtol=rtol)).sum()

    bool_arr = np.abs(x) >= 1e-15
    rel_diffs = np.abs((x[bool_arr] - y[bool_arr]) / x[bool_arr])
    if bool_arr.astype(int).sum() == 0:
        return msg + "No nonzero entries in A"
    max_rel_diff = np.max(rel_diffs)
    s = (
        msg
        + "Arrays differ in {} / {} entries. Max absolute diff: {}; max relative diff: {}".format(
            difference_count, a.size, max_diff, max_rel_diff
        )
    )

    return s


class TestSetupMethods:
    def test_0(self) -> None:
        o = setup_accelerated_solver(
            N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        )
        assert type(o) == HelmholtzSolverAccelerated


class TestHelmholtzSolverAccelerated:
    def test_0(self) -> None:
        """Tests _get_uin"""

        source_directions = torch.Tensor([0, np.pi, 3 * np.pi / 2]).to(
            SOLVER_OBJ.device
        )
        n_dirs = source_directions.shape[0]
        o = SOLVER_OBJ._get_uin(source_directions)
        assert o.shape == (n_dirs, SOLVER_OBJ.N**2)

    def test_1(self) -> None:
        # solver_obj = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        n_dirs = 3

        x = torch.randn(
            size=(
                N_PIXELS * N_PIXELS,
                n_dirs,
            )
        ).to(SOLVER_OBJ.device)

        y = SOLVER_OBJ._G_apply(x)

        assert y.shape == x.shape

    def test_2(self) -> None:
        """Tests the accelerated G apply against dense G apply for one-sparse
        input.
        """
        # solver_obj = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        print(SOLVER_OBJ.h)

        diag_correction = find_diag_correction(SOLVER_OBJ.h, SOLVER_OBJ.frequency)

        int_greens_matrix = greensfunction2(
            SOLVER_OBJ.domain_points.numpy(),
            SOLVER_OBJ.frequency,
            diag_correction=diag_correction,
            dx=SOLVER_OBJ.h,
        )

        q_0 = torch.zeros((N_PIXELS**2, 1))
        q_0[0, 0] = 1.0

        out = SOLVER_OBJ._G_apply(q_0).cpu().numpy()

        check_arrays_close(out.flatten(), int_greens_matrix[:, 0].flatten())

    def test_3(self) -> None:
        """Tests the accelerated G apply against dense G apply for random input."""
        # solver_obj = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        print(SOLVER_OBJ.h)

        diag_correction = find_diag_correction(SOLVER_OBJ.h, SOLVER_OBJ.frequency)

        int_greens_matrix = greensfunction2(
            SOLVER_OBJ.domain_points.numpy(),
            SOLVER_OBJ.frequency,
            diag_correction=diag_correction,
            dx=SOLVER_OBJ.h,
        )
        n_dirs = 3
        sigma = torch.randn(size=(N_PIXELS**2, n_dirs))

        out_a = SOLVER_OBJ._G_apply(sigma).cpu().numpy()

        out_b = int_greens_matrix @ sigma.numpy()

        check_arrays_close(out_a, out_b)

    def test_4(self) -> None:
        """Tests that the Helmholtz_solve_exterior routine returns without error
        on a single direction"""

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        z = int(N_PIXELS / 2)
        scattering_obj[z : z + 2, z : z + 2] = np.ones_like(
            scattering_obj[z : z + 2, z : z + 2]
        )

        dirs = np.array(
            [
                np.pi / 2,
            ]
        )
        out = SOLVER_OBJ.Helmholtz_solve_exterior(dirs, scattering_obj)
        assert out.shape == (
            1,
            N_PIXELS,
        )

    def test_5(self) -> None:
        """Tests Helmholtz_solve_interior on a zero scattering potential."""
        # SOLVER_OBJ = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        dirs = np.array([np.pi / 2, np.pi / 4, 3, 0])
        u_tot, u_in, u_scat = SOLVER_OBJ.Helmholtz_solve_interior(dirs, scattering_obj)

        check_arrays_close(u_tot, u_in)
        check_arrays_close(u_scat, np.zeros_like(u_scat))

    def test_6(self) -> None:
        """Tests that Helmholtz_solve_interior routine returns without error"""

        # SOLVER_OBJ = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        z = int(N_PIXELS / 2)
        scattering_obj[z : z + 2, z : z + 2] = np.ones_like(
            scattering_obj[z : z + 2, z : z + 2]
        )
        dirs = np.array([np.pi / 4, -np.pi / 4])
        n_dirs = dirs.shape[0]

        out = SOLVER_OBJ.Helmholtz_solve_interior(dirs, scattering_obj)

        for x in out:
            assert x.shape == (n_dirs, N_PIXELS, N_PIXELS)

    def test_7(self) -> None:
        """Tests that the Helmholtz_solve_exterior routine returns without error
        on multiple directions"""

        # SOLVER_OBJ = setup_accelerated_solver(
        #     N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        # )

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        z = int(N_PIXELS / 2)
        scattering_obj[z : z + 2, z : z + 2] = np.ones_like(
            scattering_obj[z : z + 2, z : z + 2]
        )

        dirs = np.array([np.pi / 2, np.pi / 3, 5 * np.pi / 2])
        n_dirs = dirs.shape[0]
        out = SOLVER_OBJ.Helmholtz_solve_exterior(dirs, scattering_obj)
        assert out.shape == (
            n_dirs,
            N_PIXELS,
        )


if __name__ == "__main__":
    pytest.main()
