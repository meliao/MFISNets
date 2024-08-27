import pytest
import torch
import time
import numpy as np
import scipy.interpolate
from src.data.data_transformations import (
    prep_conv_interp_1d,
    prep_conv_interp_2d,
    prep_polar_padder,
    apply_interp_2d,
    prep_rs_to_mh_interp,
)

class Test_precomputed_interpolation:
    # New test for the precomputed convolution filters
    def _basic_test(
        self: None, data_grid_2d, random_vals_2d, interp_grid_2d, ref_vals
    ) -> float:
        """Performs comparison for a given set of grids/values"""
        # Prepare the filters
        convx, convy = prep_conv_interp_2d(
            data_grid_2d[0],
            data_grid_2d[1],
            interp_grid_2d,
            bc_modes=("extend", "extend"),
            a_neg_half=False,
        )

        # Apply filters
        sampled_vals_conv_2d = apply_interp_2d(convx, convy, random_vals_2d)

        # Find relative error
        rel_err = np.linalg.norm(sampled_vals_conv_2d - ref_vals) / np.linalg.norm(
            ref_vals
        )
        return sampled_vals_conv_2d, rel_err

    def test_rand_uniform(self: None) -> None:
        """Compare conv vs scipy interpn on random data with uniform random sampling"""
        rng = np.random.default_rng(2)

        # Generate the grid first
        n = 100  # data grid points (each dimension)
        m = 100**2  # output interpolated points (each dimension)
        grid_x = np.linspace(0, 1, n, endpoint=True)
        grid_y = np.linspace(0, 1, n, endpoint=True)
        data_grid_2d = (grid_x, grid_y)
        random_vals_2d = rng.uniform(0, 1, size=(n, n))
        interp_grid_2d = rng.uniform(0, 1, size=(m, 2))

        # Generate reference interpolation
        sampled_vals_ref = scipy.interpolate.interpn(
            data_grid_2d,
            random_vals_2d,
            interp_grid_2d,
            method="cubic",
            bounds_error=False,
            fill_value=None,
        )

        _, rel_err = self._basic_test(
            data_grid_2d, random_vals_2d, interp_grid_2d, sampled_vals_ref
        )
        # print(f"(uniform random) relative error: {rel_err:.3e}")
        assert rel_err < 0.1

    def test_rand_disk(self: None) -> None:
        """Compare conv vs scipy interpn on random data sampled non-uniformly on a disk"""
        rng = np.random.default_rng(3)

        # Generate the grid first
        n = 100  # data grid points (each dimension)
        m = 100**2  # output interpolated points (each dimension)
        grid_x = np.linspace(0, 1, n, endpoint=True)
        grid_y = np.linspace(0, 1, n, endpoint=True)
        data_grid_2d = (grid_x, grid_y)
        random_vals_2d = rng.uniform(0, 1, size=(n, n))

        radii = rng.uniform(0, 0.5, size=m)
        angles = rng.uniform(0, 2 * np.pi, size=m)
        interp_grid_2d = (
            0.5 + np.array([radii * np.cos(angles), radii * np.sin(angles)]).T
        )

        # Generate reference interpolation
        sampled_vals_ref = scipy.interpolate.interpn(
            data_grid_2d,
            random_vals_2d,
            interp_grid_2d,
            method="cubic",
            bounds_error=False,
            fill_value=None,
        )

        _, rel_err = self._basic_test(
            data_grid_2d, random_vals_2d, interp_grid_2d, sampled_vals_ref
        )
        # rel_err = self._basic_test(data_grid_2d, random_vals_2d, interp_grid_2d)
        print(f"(disk) relative error: {rel_err:.3e}")
        assert rel_err < 0.1

    def _compare_vs_interpn_1d(
        self: None,
        data_grid: np.ndarray,
        data_vals: np.ndarray,
        xi: np.ndarray,
        interp_vals_ref: np.ndarray,
        periodic: bool = False,
    ) -> dict:
        """Helper to run tests in the 1D case
        Compares conv against scipy interpn
        Returns errors and runtimes in a dictionary
        """
        # Take care of padding
        # itvl = data_grid[1] - data_grid[0]
        pad_width = 10 if periodic else 0
        padded_data_grid = np.pad(
            data_grid, pad_width, mode="reflect", reflect_type="odd"
        )
        padded_data_vals = np.pad(data_vals, pad_width, mode="wrap")
        ref_norm = np.linalg.norm(interp_vals_ref)

        # First run scipy interpn
        interpn_start_time = time.perf_counter()
        interpn_vals = scipy.interpolate.interpn(
            padded_data_grid[np.newaxis, :],
            padded_data_vals,
            xi,
            method="cubic",
            bounds_error=False,
            fill_value=None,
        )
        interpn_run_time = time.perf_counter() - interpn_start_time
        interpn_abs_err = interpn_vals - interp_vals_ref
        interpn_err = np.linalg.norm(interpn_abs_err) / ref_norm

        # Next prepare and apply the convolution operator
        bc_mode = "periodic" if periodic else "extend"
        # print(f"Using bc mode {bc_mode}")
        conv_prep_start_time = time.perf_counter()
        conv_op = prep_conv_interp_1d(
            data_grid,
            xi,
            bc_mode=bc_mode,
            a_neg_half=True,
        )
        conv_prep_run_time = time.perf_counter() - conv_prep_start_time

        conv_apply_start_time = time.perf_counter()
        conv_vals = conv_op @ data_vals
        conv_apply_run_time = time.perf_counter() - conv_apply_start_time

        conv_abs_err = conv_vals - interp_vals_ref
        conv_err = np.linalg.norm(conv_abs_err) / ref_norm

        res_dict = {
            "interpn_vals": interpn_vals,
            "interpn_err": interpn_err,
            "conv_vals": conv_vals,
            "conv_err": conv_err,
            "interpn_run_time": interpn_run_time,
            "conv_prep_run_time": conv_prep_run_time,
            "conv_apply_run_time": conv_apply_run_time,
        }

        return res_dict

    def test_sines_pbc1d(self: None) -> None:
        """Test on sum of sines, comparing vs scipy interpn
        conv approach is allowed to have significantly higher error
        but the application time must be reasonably quick

        Note that in the 1D case, scipy interpn is quite fast so this is more of a sanity check
        """
        n = 100  # data grid points
        m = 100  # output interpolated points
        data_grid_pbc1d = np.linspace(
            0, 1, n, endpoint=False
        )  # leave out the endpoint to test "extrapolation"

        # Choose data as a sum of sines?
        rng = np.random.default_rng(3)
        interp_grid_pbc1d = np.sort(rng.uniform(0, 1, size=m))

        compt_count = 4
        # freqs = rng.choice(np.arange(1,5), size=compt_count)
        freqs = 1 + 2 * np.arange(compt_count)
        random_phase_offsets = rng.uniform(0, 1, size=compt_count)
        true_fn_pbc1d = lambda points: np.sin(
            2
            * np.pi
            * (
                freqs[np.newaxis, :] * points[:, np.newaxis]
                + random_phase_offsets[np.newaxis, :]
            )
        ).mean(1)

        # Measurements on the data grid
        data_sines_pbc1d = true_fn_pbc1d(data_grid_pbc1d)

        # Ground truth from our sinusoids
        sampled_ref_pbc1d = true_fn_pbc1d(interp_grid_pbc1d)

        res_dict = self._compare_vs_interpn_1d(
            data_grid_pbc1d, data_sines_pbc1d, interp_grid_pbc1d, sampled_ref_pbc1d
        )
        sines_interpn_err = res_dict["interpn_err"]
        sines_conv_err = res_dict["conv_err"]
        interpn_run_time = res_dict["interpn_run_time"]
        conv_prep_run_time = res_dict["conv_prep_run_time"]
        conv_apply_run_time = res_dict["conv_apply_run_time"]

        # Zoom in on the boundary/"endzone"
        endzone_grid = np.sort(rng.uniform(0.95, 1.05, size=m))
        # endzone_grid = np.linspace(0.95, 1.05, 100, endpoint=False)
        # endzone_grid = np.linspace(0.95, 1.05, 20, endpoint=False)

        endzone_true_vals = true_fn_pbc1d(endzone_grid)
        # conv_to_endzone = prep_conv_interp_1d(data_grid_pbc1d, endzone_grid, bc_mode="periodic")
        # endzone_conv_vals = conv_to_endzone @ data_sines_pbc1d
        # endzone_conv_err = np.linalg.norm(endzone_conv_vals - endzone_true_vals) \
        #     / np.linalg.norm(endzone_true_vals)
        endzone_res_dict = self._compare_vs_interpn_1d(
            data_grid_pbc1d,
            data_sines_pbc1d,
            endzone_grid,
            endzone_true_vals,
            periodic=True,
        )
        endzone_conv_err = endzone_res_dict["conv_err"]
        endzone_interpn_err = endzone_res_dict["interpn_err"]

        print("Sum-of-sines test case")
        print(
            f"Runtimes: {interpn_run_time*1000:.2f}ms (interpn), "
            f"{conv_prep_run_time*1000:.2f}ms (prep conv), "
            f"{conv_apply_run_time*1000:.2f} ms (apply conv)"
        )
        print(
            f"Relative error from scipy interpolation: {sines_interpn_err:.3e} "
            f"({endzone_interpn_err:.3e} endzone)"
        )
        print(
            f"Relative error from conv interpolation:  {sines_conv_err:.3e} "
            f"({endzone_conv_err:.3e} endzone)"
        )
        # Let the error be below a certain threshold
        # or within 100x of scipy interp in case it's a tough case
        assert sines_conv_err <= max(100 * sines_interpn_err, 2e-3)
        assert endzone_conv_err <= max(100 * endzone_interpn_err, 2e-3)

    def test_gaussian_1d(self: None) -> None:
        """Test on a sum of gaussians, comparing vs scipy interpn
        conv approach is allowed to have significantly higher error
        but the application time must be reasonably quick

        Note that in the 1D case, scipy interpn is quite fast so this is more of a sanity check
        """
        rng = np.random.default_rng(1)
        n = 100  # data grid points
        m = 100  # output interpolated points
        data_grid_1d = np.linspace(
            0, 1, n, endpoint=False
        )  # leave out the endpoint to test "extrapolation"

        # Choose data as a sum of sines?
        interp_grid_1d = np.sort(rng.uniform(0, 1, size=m))

        gauss_count = 3  # number of gaussians
        sigs = np.clip(
            np.abs(rng.normal(loc=0.1, scale=0.05, size=gauss_count)), 0.05, 0.15
        )
        centers = rng.uniform(0.2, 0.8, size=gauss_count)

        def sum_of_gaussians(points):
            denom = np.sqrt(2 * np.pi) * sigs[np.newaxis, :]
            scaled_offsets = (points[:, np.newaxis] - centers[np.newaxis, :]) / sigs[
                np.newaxis, :
            ]
            return (np.exp(-0.5 * scaled_offsets**2) / denom).mean(1)

        # Measurements on the data grid
        data_gauss_1d = sum_of_gaussians(data_grid_1d)

        # Ground truth from our gaussians
        sampled_ref_gauss_1d = sum_of_gaussians(interp_grid_1d)
        ref_norm = np.linalg.norm(sampled_ref_gauss_1d)

        res_dict = self._compare_vs_interpn_1d(
            data_grid_1d, data_gauss_1d, interp_grid_1d, sampled_ref_gauss_1d
        )
        gauss_interpn_err = res_dict["interpn_err"]
        gauss_conv_err = res_dict["conv_err"]
        interpn_run_time = res_dict["interpn_run_time"]
        conv_prep_run_time = res_dict["conv_prep_run_time"]
        conv_apply_run_time = res_dict["conv_apply_run_time"]

        print("Sum-of-gaussians test case")
        print(
            f"Runtimes: {interpn_run_time*1000:.2f}ms (interpn), "
            f"{conv_prep_run_time*1000:.2f}ms (prep conv), "
            f"{conv_apply_run_time*1000:.2f} ms (apply conv)"
        )
        print(f"Relative error from scipy interpolation: {gauss_interpn_err:.3e}")
        print(f"Relative error from conv interpolation:  {gauss_conv_err:.3e}")

        # Let the error be below a certain threshold
        # or within 100x of scipy interp in case it's a tough case
        assert gauss_conv_err <= max(100 * gauss_interpn_err, 2e-3)

    def _compare_vs_interpn_2d(
        self: None,
        data_grid_x: np.ndarray,
        data_grid_y: np.ndarray,
        data_vals: np.ndarray,
        xi: np.ndarray,
        interp_vals_ref: np.ndarray,
        from_polar: bool = False,
    ) -> dict:
        """Helper to run tests in the 2D case
        Compares conv against scipy interpn
        Returns errors and runtimes in a dictionary

        from_polar flag lets us know if padding is needed; treats the x-dim as radius and y-dim as angle
        """
        # Take care of padding
        nx = data_grid_x.shape[0]
        ny = data_grid_y.shape[0]
        ref_norm = np.linalg.norm(interp_vals_ref)
        if from_polar:
            pad_rad = 1
            pad_width = 10
            padded_data_grid_y = np.pad(
                data_grid_y, pad_width, mode="reflect", reflect_type="odd"
            )
            padded_data_grid_x, polar_padder = prep_polar_padder(
                data_grid_x,
                ny,
                dim=0,
            )
            # padded_data_vals = polar_padder(data_vals.reshape(nx, ny))
        else:
            pad_rad = 0
            pad_width = 0
            padded_data_grid_x = data_grid_x
            padded_data_grid_y = data_grid_y
            # padded_data_vals = data_vals.reshape(nx, ny)
            polar_padder = lambda x: x

        # First run scipy interpn
        interpn_start_time = time.perf_counter()
        padded_data_vals = polar_padder(data_vals.reshape(nx, ny))
        if from_polar:
            # pad it extra for the periodic condition
            padded_data_vals = np.pad(
                padded_data_vals, ((0, 0), (pad_width, pad_width)), mode="wrap"
            )
        interpn_vals = scipy.interpolate.interpn(
            (padded_data_grid_x, padded_data_grid_y),
            padded_data_vals,
            xi,
            method="cubic",
            bounds_error=False,
            fill_value=None,
        )
        interpn_run_time = time.perf_counter() - interpn_start_time
        interpn_abs_err = interpn_vals - interp_vals_ref
        interpn_err = np.linalg.norm(interpn_abs_err) / ref_norm

        # Next prepare and apply the convolution operator
        bc_mode_x = "extend"
        bc_mode_y = "periodic" if from_polar else "extend"
        conv_prep_start_time = time.perf_counter()
        conv_op_x, conv_op_y = prep_conv_interp_2d(
            padded_data_grid_x,
            data_grid_y,
            xi,
            bc_modes=(bc_mode_x, bc_mode_y),
            a_neg_half=True,
        )
        conv_prep_run_time = time.perf_counter() - conv_prep_start_time

        conv_apply_start_time = time.perf_counter()
        padded_data_vals = polar_padder(data_vals.reshape(nx, ny))
        conv_vals = apply_interp_2d(
            conv_op_x, conv_op_y, padded_data_vals.reshape(nx + pad_rad, ny)
        )
        conv_apply_run_time = time.perf_counter() - conv_apply_start_time

        conv_abs_err = conv_vals - interp_vals_ref
        conv_err = np.linalg.norm(conv_abs_err) / ref_norm
        res_dict = {
            "interpn_vals": interpn_vals,
            "interpn_err": interpn_err,
            "conv_vals": conv_vals,
            "conv_err": conv_err,
            "interpn_run_time": interpn_run_time,
            "conv_prep_run_time": conv_prep_run_time,
            "conv_apply_run_time": conv_apply_run_time,
        }
        return res_dict

    def test_cart_to_polar(self: None) -> None:
        """2D case: resample gaussians from cartesian to polar grids"""
        nx = 200  # data grid points
        ny = 199  # data grid points
        period = 1  # 2*np.pi
        data_grid_x = np.linspace(
            -period / 2, period / 2, nx, endpoint=False
        )  # leave out the endpoint to test "extrapolation"
        data_grid_y = np.linspace(
            -period / 2, period / 2, ny, endpoint=False
        )  # leave out the endpoint to test "extrapolation"
        data_grid_xy = np.array(np.meshgrid(data_grid_x, data_grid_y)).T.reshape(
            nx * ny, 2
        )

        # Choose data as a sum of sines?
        rng = np.random.default_rng(11 + 2)
        # interp_grid_2d = np.sort(rng.uniform(-period/2, period/2, size=(m,2)))

        g2_count = 10  # number of gaussians
        # for simplicity pick a diagonal covariance matrix by splitting each dimension
        sigs_2d = np.clip(
            np.abs(rng.normal(loc=0.05, scale=0.03, size=(g2_count, 2))), 0.01, 0.10
        )
        centers_2d = rng.normal(loc=0, scale=0.3 * period / 2, size=(g2_count, 2))

        # Rotate to show that the separability of the filter is not entirely trivial
        rot_mat = 1 / np.sqrt(2) * np.array([[1, -1], [1, 1]])

        def sog2_fn(points):
            """Sum-of-gaussians 2D function"""
            points = points @ rot_mat.T
            denom = np.sqrt(2 * np.pi) * np.sqrt(
                np.prod(sigs_2d[np.newaxis, :, :], axis=-1)
            )
            # denom = 1
            scaled_offsets = (
                points[:, np.newaxis, :] - centers_2d[np.newaxis, :, :]
            ) / sigs_2d[np.newaxis, :, :]
            return (np.exp(-0.5 * (scaled_offsets**2).sum(2)) / denom).mean(1)

        # Measurements on the data grid
        data_gauss_2d = sog2_fn(data_grid_xy)

        # Target polar grid points
        nr = 100  # 200
        ntheta = 98  # 198
        center = np.array([0, 0])
        polar_grid_r = np.linspace(0, 0.6 * period, nr, endpoint=False)
        polar_grid_theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        polar_grid_rtheta = np.array(
            np.meshgrid(polar_grid_r, polar_grid_theta)
        ).T.reshape(nr * ntheta, 2)

        # Find the polar grid points in cartesian coords
        polar_grid_xy = np.array(
            [
                center[0] + polar_grid_rtheta[:, 0] * np.cos(polar_grid_rtheta[:, 1]),
                center[1] + polar_grid_rtheta[:, 0] * np.sin(polar_grid_rtheta[:, 1]),
            ]
        ).T

        polar_data_ref = sog2_fn(polar_grid_xy).reshape(nr, ntheta)
        polar_ref_norm = np.linalg.norm(polar_data_ref)

        res_dict = self._compare_vs_interpn_2d(
            data_grid_x,
            data_grid_y,
            data_gauss_2d,
            polar_grid_xy,
            polar_data_ref.reshape(nr * ntheta),
        )

        gauss_interpn_err = res_dict["interpn_err"]
        gauss_conv_err = res_dict["conv_err"]
        interpn_run_time = res_dict["interpn_run_time"]
        conv_prep_run_time = res_dict["conv_prep_run_time"]
        conv_apply_run_time = res_dict["conv_apply_run_time"]

        # Note: the conv application scales poorly with (nr, ntheta)...
        # nr*ntheta = 100**2 -> 13 ms application
        # nr*ntheta = 4 * 100**2 -> 70-80 ms application
        # Speedup falls from 98x to 73x
        # I think this is partly because we need to deal with the output points
        # twice -- once for each dimension

        print("Cartesian-to-polar test case (with sum-of-gaussians)")
        print(
            f"Runtimes: {interpn_run_time*1000:.2f}ms (interpn), "
            f"{conv_prep_run_time*1000:.2f}ms (prep conv), "
            f"{conv_apply_run_time*1000:.2f} ms (apply conv)"
        )
        print(f"Conv application speedup: {interpn_run_time/conv_apply_run_time:.0f}x")
        print(f"Relative error from scipy interpolation: {gauss_interpn_err:.3e}")
        print(f"Relative error from conv interpolation:  {gauss_conv_err:.3e}")

        # Let the error be below a certain threshold
        # or within 10x of scipy interp in case it's a tough case
        assert gauss_conv_err <= max(20 * gauss_interpn_err, 5e-3)

        # # Require at least 50x speedup..
        # # Currently, conv_prep_run_time ~= interp_run_time
        # assert conv_prep_run_time <= 10 * interpn_run_time
        # assert 50 * conv_apply_run_time <= interpn_run_time

    def test_polar_to_cart(self: None) -> None:
        """2D case: resample gaussians from polar to cartesian grids"""
        # TODO second
        nx = 200  # data grid points
        ny = 199  # data grid points
        period = 1  # 2*np.pi
        data_grid_x = np.linspace(
            -period / 2, period / 2, nx, endpoint=False
        )  # leave out the endpoint to test "extrapolation"
        data_grid_y = np.linspace(
            -period / 2, period / 2, ny, endpoint=False
        )  # leave out the endpoint to test "extrapolation"
        data_grid_xy = np.array(np.meshgrid(data_grid_x, data_grid_y)).T.reshape(
            nx * ny, 2
        )

        # Choose data as a sum of sines?
        rng = np.random.default_rng(11 + 2)

        g2_count = 10  # number of gaussians
        # for simplicity pick a diagonal covariance matrix by splitting each dimension
        sigs_2d = np.clip(
            np.abs(rng.normal(loc=0.05, scale=0.03, size=(g2_count, 2))), 0.01, 0.10
        )
        centers_2d = rng.normal(loc=0, scale=0.3 * period / 2, size=(g2_count, 2))

        # Rotate to show that the separability of the filter is not entirely trivial
        rot_mat = 1 / np.sqrt(2) * np.array([[1, -1], [1, 1]])

        def sog2_fn(points):
            """Sum-of-gaussians 2D function"""
            points = points @ rot_mat.T
            denom = np.sqrt(2 * np.pi) * np.sqrt(
                np.prod(sigs_2d[np.newaxis, :, :], axis=-1)
            )
            # denom = 1
            scaled_offsets = (
                points[:, np.newaxis, :] - centers_2d[np.newaxis, :, :]
            ) / sigs_2d[np.newaxis, :, :]
            return (np.exp(-0.5 * (scaled_offsets**2).sum(2)) / denom).mean(1)

        # Measurements on the data grid
        data_gauss_2d = sog2_fn(data_grid_xy)

        # Target polar grid points
        nr = 100  # 200
        ntheta = 98  # 198
        center = np.array([0, 0])
        polar_grid_r = np.linspace(0, 0.6 * period, nr, endpoint=False)
        polar_grid_theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        polar_grid_rtheta = np.array(
            np.meshgrid(polar_grid_r, polar_grid_theta)
        ).T.reshape(nr * ntheta, 2)

        # Find the polar grid points in cartesian coords
        polar_grid_xy = np.array(
            [
                center[0] + polar_grid_rtheta[:, 0] * np.cos(polar_grid_rtheta[:, 1]),
                center[1] + polar_grid_rtheta[:, 0] * np.sin(polar_grid_rtheta[:, 1]),
            ]
        ).T

        polar_data_ref = sog2_fn(polar_grid_xy).reshape(nr, ntheta)
        polar_ref_norm = np.linalg.norm(polar_data_ref)

        # Put the cartesian grid into polar coordinates
        cart_grid_radii = np.sqrt(data_grid_xy[:, 0] ** 2 + data_grid_xy[:, 1] ** 2)
        cart_grid_thetas = np.mod(
            np.arctan2(data_grid_xy[:, 1] - center[1], data_grid_xy[:, 0] - center[0]),
            2 * np.pi,
        )
        cart_grid_polar_coords = np.array([cart_grid_radii, cart_grid_thetas]).T

        res_dict = self._compare_vs_interpn_2d(
            polar_grid_r,
            polar_grid_theta,
            polar_data_ref,
            cart_grid_polar_coords,
            data_gauss_2d.reshape(nx * ny),
            from_polar=True,
        )

        gauss_interpn_err = res_dict["interpn_err"]
        gauss_conv_err = res_dict["conv_err"]
        interpn_run_time = res_dict["interpn_run_time"]
        conv_prep_run_time = res_dict["conv_prep_run_time"]
        conv_apply_run_time = res_dict["conv_apply_run_time"]

        print("Polar-to-Cartesian test case (with sum-of-gaussians)")
        print(
            f"Runtimes: {interpn_run_time*1000:.2f}ms (interpn), "
            f"{conv_prep_run_time*1000:.2f}ms (prep conv), "
            f"{conv_apply_run_time*1000:.2f} ms (apply conv)"
        )
        print(f"Conv application speedup: {interpn_run_time/conv_apply_run_time:.0f}x")
        print(f"Relative error from scipy interpolation: {gauss_interpn_err:.3e}")
        print(f"Relative error from conv interpolation:  {gauss_conv_err:.3e}")

        # Let the error be below a certain threshold
        # or within 10x of scipy interp in case it's a tough case
        assert gauss_conv_err <= max(20 * gauss_interpn_err, 5e-3)

    def test_rs_to_mh(self: None) -> None:
        """Convert between (r, s) and (m, h)
        Note that this implies periodic boundary conditions
        Use sum of sines in 2D

        Compare against the code used in the original `data_preprocess_pipeline`
        """
        nr = 100  # data grid points
        ns = 80  # data grid points
        period = 2 * np.pi
        data_grid_r = np.linspace(0, period, nr, endpoint=False)
        data_grid_s = np.linspace(0, period, ns, endpoint=False)
        data_grid_rs = np.array(np.meshgrid(data_grid_r, data_grid_s)).T.reshape(
            nr * ns, 2
        )

        rng = np.random.default_rng(11 + 3)
        g2_count = 20  # number of gaussians

        freqs_2d = rng.integers(1, 10, size=(g2_count, 2))
        offsets_2d = rng.uniform(0, 2 * np.pi, size=(g2_count, 2))

        def sos2_pbc_fn(points):
            """sum-of-sines 2D function"""
            points = np.mod(points, period)
            phase_vals = (
                freqs_2d[np.newaxis, :, :] * points[:, np.newaxis, :]
                + offsets_2d[np.newaxis, :, :]
            ).sum(2)
            return np.sin(phase_vals).mean(1)

        # Measurements on the data grid
        data_sines_pbc2d = sos2_pbc_fn(data_grid_rs)

        # Now construct the target grid points
        nm = 200
        nh = 47
        grid_m = np.linspace(0, period, nm, endpoint=False)
        grid_h = np.linspace(-period / 4, period / 4, nh, endpoint=False)
        grid_mh = np.array(np.meshgrid(grid_m, grid_h)).T.reshape(nm * nh, 2)

        # Find the target grid points in the old coordinate system
        grid_mh_in_rs_coords = np.array(
            [
                np.mod(grid_mh[:, 0] + grid_mh[:, 1], period),  # r
                np.mod(grid_mh[:, 0] - grid_mh[:, 1], period),  # s
            ]
        ).T

        grid_mm, grid_hh = np.meshgrid(grid_m, grid_h)
        grid_mh_2 = np.vstack((grid_mm.flatten(), grid_hh.flatten())).T

        # I've defined grid_mh as roughly the transpose of Owen's version
        # This way, I can reshape with (nm, nh) in the same order as (m, h)
        # and then just take the transpose when displaying...
        assert grid_mh.shape == grid_mh_2.shape
        assert np.allclose(
            grid_mh, grid_mh_2.reshape(nh, nm, 2).transpose(1, 0, 2).reshape(nm * nh, 2)
        )
        sampled_ref_sines_pbc2d = sos2_pbc_fn(grid_mh_in_rs_coords)
        ref_norm = np.linalg.norm(sampled_ref_sines_pbc2d)

        # Run the scipy interpolator
        start_time = time.perf_counter()
        pad_width = 10
        padded_data_grid_r = np.pad(
            data_grid_r, pad_width, mode="reflect", reflect_type="odd"
        )
        padded_data_grid_s = np.pad(
            data_grid_s, pad_width, mode="reflect", reflect_type="odd"
        )
        padded_data_sines_pbc2d = np.pad(
            data_sines_pbc2d.reshape(nr, ns), pad_width, mode="wrap"
        )

        mh_sines_scipy_vals = scipy.interpolate.interpn(
            (padded_data_grid_r, padded_data_grid_s),
            padded_data_sines_pbc2d,
            grid_mh_in_rs_coords,
            method="cubic",
            bounds_error=False,
            fill_value=None,
        )
        interpn_run_time = time.perf_counter() - start_time

        sines_interpn_err = (
            np.linalg.norm(mh_sines_scipy_vals - sampled_ref_sines_pbc2d) / ref_norm
        )

        # Run my conv interpolator
        start_time = time.perf_counter()
        conv_rs_to_m = prep_conv_interp_1d(
            data_grid_r, grid_mh_in_rs_coords[:, 0], bc_mode="periodic"
        )
        conv_rs_to_h = prep_conv_interp_1d(
            data_grid_s, grid_mh_in_rs_coords[:, 1], bc_mode="periodic"
        )
        conv_prep_run_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        conv_mh_vals = apply_interp_2d(
            conv_rs_to_m, conv_rs_to_h, data_sines_pbc2d.reshape(nr, ns)
        )
        conv_apply_run_time = time.perf_counter() - start_time

        mh_sines_conv_err = (
            np.linalg.norm(conv_mh_vals - sampled_ref_sines_pbc2d) / ref_norm
        )

        # Now compare against `prep_rs_to_mh_interp`...
        conv_to_m, conv_to_h = prep_rs_to_mh_interp(
            data_grid_r, data_grid_s, nm, nh, a_neg_half=True
        )

        # Before reporting the summary, make sure I'm using the same operators
        # as what `prep_rs_to_mh_interp` gives
        assert np.allclose(conv_to_m.todense(), conv_rs_to_m.todense())
        assert np.allclose(conv_to_h.todense(), conv_rs_to_h.todense())

        print("rs-to-mh test case (with sum-of-sines)")
        print(
            f"Runtimes: {interpn_run_time*1000:.2f}ms (interpn), "
            f"{conv_prep_run_time*1000:.2f}ms (prep conv), "
            f"{conv_apply_run_time*1000:.2f} ms (apply conv)"
        )
        print(f"Conv application speedup: {interpn_run_time/conv_apply_run_time:.0f}x")
        print(f"Relative error from interpolation via scipy: {sines_interpn_err:.3e}")
        print(f"Relative error from interpolation via conv: {mh_sines_conv_err:.5e}")

        # Let the error be below a certain threshold
        # or within 10x of scipy interp in case it's a tough case
        assert mh_sines_conv_err <= max(20 * sines_interpn_err, 5e-3)

if __name__ == "__main__":
    pytest.main()
