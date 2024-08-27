import numpy as np
import pytest

from solvers.integral_equation.HelmholtzSolver import (
    HelmholtzSolverAccelerated,
    setup_accelerated_solver,
)

N_PIXELS = 50
SPATIAL_DOMAIN_MAX = 0.5
WAVENUMBER = 16
RECEIVER_RADIUS = 100


class TestAAA:
    def test_0(self) -> None:
        """Tests that the single_Helmholtz_solve_interior routine returns without error"""

        solver_obj = setup_accelerated_solver(
            N_PIXELS, SPATIAL_DOMAIN_MAX, WAVENUMBER, RECEIVER_RADIUS
        )

        scattering_obj = np.zeros((N_PIXELS, N_PIXELS))
        z = int(N_PIXELS / 2)
        scattering_obj[z : z + 2, z : z + 2] = np.ones_like(
            scattering_obj[z : z + 2, z : z + 2]
        )

        dirs = np.array([np.pi / 4, np.pi / 6, 0])

        N_DIRS = dirs.shape[0]
        out = solver_obj.Helmholtz_solve_interior(dirs, scattering_obj)

        for x in out:
            assert x.shape == (N_DIRS, N_PIXELS, N_PIXELS)


if __name__ == "__main__":
    pytest.main()
