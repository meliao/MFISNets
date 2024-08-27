# Helmholtz equation solver
# Sets up a solver object for a given domain and incoming wave frequency
# Solves an integral equation formulation of the Helmholtz equation
# using gmres.

import logging
import numpy as np

import torch
import cola

from typing import Tuple
from scipy.sparse.linalg import LinearOperator, gmres

from solvers.integral_equation.Helmholtz_solver_utils import (
    greensfunction3,
    getGscat2circ,
    find_diag_correction,
    get_extended_grid,
)

logging.getLogger("cola-ml").setLevel(logging.WARNING)
logging.getLogger("cola").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("numpy").setLevel(logging.WARNING)
logging.getLogger("plum").setLevel(logging.WARNING)
logging.getLogger("plum-dispatch").setLevel(logging.WARNING)
logging.getLogger("plum-dispatch").setLevel(logging.WARNING)


class HelmholtzSolverBase:
    def __init__(
        self,
        domain_points: np.ndarray,
        frequency: float,
        exterior_greens_function: np.ndarray,
        N: int,
        source_dirs: np.ndarray,
        x_vals: np.ndarray,
    ) -> None:
        self.domain_points = torch.from_numpy(domain_points).to(torch.float)
        self.frequency = frequency
        self.frequency_torch = torch.Tensor([frequency]).to(torch.float)

        self.N = N
        self.source_dirs = source_dirs
        self.x_vals = x_vals
        self.domain_points_arr = domain_points.reshape((N, N, 2))

        self.h = self.domain_points_arr[0, 1, 0] - self.domain_points_arr[0, 0, 0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exterior_greens_function = (
            torch.from_numpy(exterior_greens_function).to(torch.cfloat).to(self.device)
        )

    def _get_uin(self, source_directions: torch.Tensor) -> torch.Tensor:
        """Returns a plane wave e^{ik<x,s>} sampled at the points x listed in self.domain_points_arr.
        In this equation, k is the angular frequency = self.frequency, and s is the unit vector pointing in
        direction specified by source_directions, in radiana.

        Args:
            source_directions (torch.Tensor): Has shape (n_directions,)

        Returns:
            torch.Tensor: Has shape (n_directions, self.N**2)
        """
        inc = torch.stack(
            [torch.cos(source_directions), torch.sin(source_directions)]
        ).to(torch.float)
        # print("_get_uin: inc shape: ", inc.shape)
        inner_prods = self.domain_points.to(self.device) @ inc
        # print("_get_uin: inner_prods shape: ", inner_prods.shape)

        uin = (
            torch.exp(1j * self.frequency * inner_prods).to(torch.cfloat).permute(1, 0)
        )
        return uin

    def _get_uin_sigma(
        self, source_directions: torch.Tensor, scattering_obj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """For a batch of source directions, this function generates the incoming
        plane waves uin, and also generates solutions to the integral equation

        int_{x in \Omega} (I + k^2 diag(q) G) sigma = -k^2 uin q

        Args:
            source_direction (torch.Tensor): Has shape (n_directions,)
            scattering_obj (torch.Tensor): shape (N, N)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: First output is uin which has shape
            (n_directions, N**2). Second output is sigma, which has shape (n_directions, N**2)
        """
        # print("_get_uin_sigma: scattering_obj shape", scattering_obj.shape)

        uin = self._get_uin(source_directions)

        # print("_get_uin_sigma: uin shape: ", uin.shape)

        if torch.all(scattering_obj == torch.zeros_like(scattering_obj)):
            sigma = torch.zeros((source_directions.shape[0], self.N**2))

        else:
            sigma = self._gmres_Helmholtz_inv(scattering_obj, uin)
        return uin, sigma

    def _G_apply(self, x: np.ndarray) -> np.ndarray:
        """Apply the green's function operator. This is used as a subroutine in
        GMRES calls.

        Must be implemented by a child class.

        Args:
            x (np.ndarray): Has shape (N ** 2,)

        Raises:
            NotImplementedError: This must be implemented by child classes

        Returns:
            np.ndarray: Has shape (N ** 2,)
        """
        raise NotImplementedError()

    def _gmres_Helmholtz_inv(
        self,
        scattering_obj: np.ndarray,
        uin: np.ndarray,
    ) -> np.ndarray:
        """

        Generates a solution to the integral equation

        int_{x in \Omega} (I + k^2 diag(q) G) sigma = -k^2 uin q

        Args:
            scattering_obj (np.ndarray): Has shape (N, N)
            uin (np.ndarray): Has shape (n_directions, self.N**2)
            device (torch.cuda.Device): function expects scattering_obj, u_in
            to be on this device and will compute on this device.

        Returns:
            np.ndarray: Has shape (n_directions, self.N**2)
        """
        # print(
        #     "_gmres_Helmholtz_inv: scattering_obj shape and device",
        #     scattering_obj.shape,
        #     scattering_obj.device,
        # )
        # print("_gmres_Helmholtz_inv: uin shape and device", uin.shape, uin.device)

        n = scattering_obj.shape[0]
        q = scattering_obj.flatten().unsqueeze(-1)
        # print("_gmres_Helmholtz_inv: q shape", q.shape)

        def _matvec(x: torch.Tensor) -> torch.Tensor:
            # print("_matvec: input shape: ", x.shape)
            gout = self._G_apply(x)
            term2 = (self.frequency**2) * q * gout
            # print("_matvec: term2 shape and device: ", term2.shape, term2.device)
            y = x + term2.to(torch.cfloat)
            # print("_matvec: output device: ", y.device)
            # print("_matvec: output shape: ", y.shape)
            return y

        A = cola.ops.LinearOperator(
            torch.complex64,
            (self.N**2, self.N**2),
            matmat=_matvec,
        )

        # X = A.to(self.device)
        A.device = self.device
        # print("_gmres_Helmholtz_inv: new A operator on device: ", A.device)
        b = -(self.frequency**2) * q * uin.permute(1, 0)
        b = b.to(torch.cfloat)
        # print("_gmres_Helmholtz_inv: b shape: ", b.shape)
        # print("_gmres_Helmholtz_inv: b device: ", b.device)

        sigma, out_info = cola.algorithms.gmres(A, b)

        out = sigma.permute(1, 0)

        # logging.warning(
        #     f"_gmres_Helmholtz_inv: GMRES exited after {out_info['iterations']} iterations"
        #     f" with a final error of {out_info['errors'][-1]:.4e}"  # Is this the right error?
        # )
        # print("_gmres_Helmholtz_inv: output shape: ", out.shape)
        return out

    def Helmholtz_solve_exterior(
        self, source_directions: np.ndarray, scattering_obj: np.ndarray
    ) -> np.ndarray:
        """Solve the Helmholtz equation on the exterior ring for a given source
        direction and a given scattering object.

        Returns the scattered wave field.

        Args:
            source_directions (np.ndarray): Angle in radians
            scattering_obj (np.ndarray): Has shape (N, N)

        Returns:
            np.ndarray: Has shape (N,)
        """
        directions_torch = torch.from_numpy(source_directions).to(self.device)
        scattering_obj_torch = torch.from_numpy(scattering_obj).to(self.device)
        _, sigma = self._get_uin_sigma(directions_torch, scattering_obj_torch)
        # print("Helmholtz_solve_exterior: sigma shape: ", sigma.shape)
        # print(
        #     "Helmholtz_solve_exterior: exterior_greens_function shape: ",
        #     self.exterior_greens_function.shape,
        # )

        FP = self.exterior_greens_function @ sigma.permute(1, 0)
        # FP = np.reshape(FP, (1, -1))

        return FP.permute(1, 0).cpu().numpy()

    def Helmholtz_solve_interior(
        self, source_directions: np.ndarray, scattering_obj: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solves the Helmholtz equation on the scattering domain.
        Returns the total wave field, the scattered wave field, and the incident wave field.

        Args:
            source_direction (float): _description_
            scattering_obj (np.ndarray): Has shape (N, N)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (u_tot, u_in, u_scat).
            Each has shape (N, N)
        """
        n_directions = source_directions.shape[0]
        out_shape = (n_directions, self.N, self.N)
        directions_torch = torch.from_numpy(source_directions).to(self.device)
        scattering_obj_torch = torch.from_numpy(scattering_obj).to(self.device)
        uin, sigma = self._get_uin_sigma(directions_torch, scattering_obj_torch)
        # print("Helmholtz_solve_interior: uin shape: ", uin.shape)
        # print("Helmholtz_solve_interior: sigma shape: ", sigma.shape)

        u_s = self._G_apply(sigma.permute(1, 0)).permute(1, 0)
        # print("Helmholtz_solve_interior: u_s shape: ", u_s.shape)
        # print("Uin shape: ", uin.shape)
        # print("U_s shape: ", u_s.shape)
        u_tot = u_s + uin

        return (
            u_tot.reshape(out_shape).cpu().numpy(),
            uin.reshape(out_shape).cpu().numpy(),
            u_s.reshape(out_shape).cpu().numpy(),
        )


class HelmholtzSolverAccelerated(HelmholtzSolverBase):
    def __init__(
        self,
        domain_points: np.ndarray,
        extended_domain_points: np.ndarray,
        G_fft: np.ndarray,
        frequency: float,
        exterior_greens_function: np.ndarray,
        N: int,
        source_dirs: np.ndarray,
        x_vals: np.ndarray,
    ) -> None:
        super().__init__(
            domain_points, frequency, exterior_greens_function, N, source_dirs, x_vals
        )
        self.extended_domain_points = extended_domain_points
        self.G_fft = torch.from_numpy(G_fft).to(self.device)

    def _G_apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applys the Green's function by
        1. copying x to a larger grid padded with zeros
        2. computing the 2D Fourier transform of x
        3. pointwise multiplying with the 2D FT of the Green's function.
        4. inverting the Fourier transform and undoing the padding step.

        Args:
            x (torch.Tensor): Has shape (self.N**2, n_dirs)

        Returns:
            torch.Tensor: Has shape (self.N**2, n_dirs)
        """
        # print("_fast_G_apply: input shape: ", x.shape)
        # print("_fast_G_apply: G_fft shape: ", self.G_fft.shape)
        x_shape = x.shape
        x_square = x.reshape(self.N, self.N, x_shape[1])
        x_pad = self._zero_pad(x_square, self.G_fft.shape[0])
        # print("_fast_G_apply: x_pad shape: ", x_pad.shape)

        x_fft = torch.fft.fft2(x_pad, dim=(0, 1))

        prod = torch.einsum("ab,abc->abc", self.G_fft, x_fft)
        # print("_fast_G_apply: prod shape", prod.shape)
        out_fft = torch.fft.ifft2(prod, dim=(0, 1))

        out = out_fft[: self.N, : self.N]

        o = out.reshape(x_shape)
        # logging.debug("_fast_G_apply: output shape: %s", o.shape)
        # print("_fast_G_apply: output shape: ", o.shape)

        return o

    def _zero_pad(self, v: torch.Tensor, n: int) -> torch.Tensor:
        """v has shape (n_small, n_small, n_dirs) and output has shape (n, n, n_dirs)"""
        o = torch.zeros((n, n, v.shape[2]), dtype=v.dtype, device=self.device)
        o[: v.shape[0], : v.shape[1]] = v

        return o


def setup_accelerated_solver(
    n_pixels: int,
    spatial_domain_max: float,
    wavenumber: float,
    receiver_radius: float,
    diag_correction: bool = True,
) -> HelmholtzSolverAccelerated:
    """
    Sets up a HelmholtzSolver object, where the application of the Green's function
    is accelerated by 2D FFTs. The inputs to this function describe the geometry of the
    spatial regime, as well as the wavenumber of the incident waves.

    Args:
        n_pixels (int): The number of spatial points along each axis of the scattering domain.
            Also used as the number of source/reciever directions.
        spatial_domain_max (float): Half side-length of the spatial domain centered at the origin.
        wavenumber (float): The angular wavenumber of the incident waves. The incident waves will
            have frequency 2pi*wavenumber.
        receiver_radius (float): Recievers are placed equispaced on a ring centered at the origin with this radius
        diag_correction (bool, optional): If True, a correction for the Green's function's singularity at the origin is computed.
        Setting this to True means taking extra time at startup to compute this correction, but the resulting solver is MUCH more accurate. Defaults to True.

    Returns:
        HelmholtzSolverAccelerated: A PDE solver object with all the necessary objects pre-computed.
    """

    frequency = 2 * np.pi * wavenumber
    source_receiver_directions = np.linspace(0, 2 * np.pi, n_pixels + 1)[:n_pixels]

    x = np.linspace(
        -spatial_domain_max, spatial_domain_max, num=n_pixels, endpoint=False
    )
    y = np.linspace(
        -spatial_domain_max, spatial_domain_max, num=n_pixels, endpoint=False
    )
    h = x[1] - x[0]

    X, Y = np.meshgrid(x, y)
    domain_points_lst = np.array([X.flatten(), Y.flatten()]).T

    extended_domain_points_grid = get_extended_grid(n_pixels, h)

    receiver_points = (
        receiver_radius
        * np.array(
            [np.cos(source_receiver_directions), np.sin(source_receiver_directions)]
        ).T
    )

    logging.debug(
        "precompute_objects: receiver_points shape: %s", receiver_points.shape
    )

    if diag_correction:
        diag_correction_val = find_diag_correction(h, frequency)
    else:
        diag_correction_val = None
    G_int = greensfunction3(
        extended_domain_points_grid,
        frequency,
        diag_correction=diag_correction_val,
        dx=h,
    )
    G_int_fft = np.fft.fft2(G_int)
    # interior_greens_function = greensfunction2(domain_points_lst, frequency)

    exterior_greens_function = getGscat2circ(
        domain_points_lst, receiver_points, frequency, dx=h
    )
    # exterior_greens_function = None

    out = HelmholtzSolverAccelerated(
        domain_points_lst,
        extended_domain_points_grid,
        G_int_fft,
        frequency,
        exterior_greens_function,
        n_pixels,
        source_receiver_directions,
        x,
    )
    return out
