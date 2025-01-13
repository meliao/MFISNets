"""
This file implements the forward and inverse neural networks for wave
scattering described in Fan and Ying "Solving Inverse Wave Scattering with
Deep Learning" 2019. We are calling it FYNet

The modifications to the architecture are instead of using BCR-Net, we use more
1D convolutions.
"""

import torch
import numpy as np
from src.data.data_transformations import (
    prep_polar_padder,
    prep_conv_interp_2d,
    polar_pad_and_apply,
)
from src.utils.conv_ops import (
    conv_in_fourier_space,
    apply_conv_with_polar_padding,
)

from typing import Tuple
import logging

class FYNetInverse(torch.nn.Module):
    def __init__(
        self,
        N_h: int,
        N_rho: int,
        c_1d: int,
        c_2d: int,
        w_1d: int,
        w_2d: int,
        N_cnn_1d: int,
        N_cnn_2d: int,
        x_vals: np.ndarray = None,
        y_vals: np.ndarray = None,
        rho_vals: np.ndarray = None,
        theta_vals: np.ndarray = None,
        output_as_cart: bool = False,
        skip_conv_calc: bool = False,
        init_mode: str = None,
    ) -> None:
        """The inverse NN described in section 2 of FY19. NN inputs have shape
        (batch, N_M, N_H, 2) and outputs have shape (batch, N_theta, N_rho),
        and we assume N_M == N_theta.

        Args:
            N_h  (int): Number of h grid points in (m, h) coordinates
            N_rho (int): Number of radial grid points in polar coordinates
            c_1d (int): Number of channels for 1d conv
            c_2d (int): Number of channels for 2d conv
            w_1d (int): Width of the 1d conv kernel
            w_2d (int): Width of the 2d conv kernel
            N_cnn_1d (int): Number of 1d conv layers
            N_cnn_2d (int): Number of 2d conv layers

            x_vals (np ndarray): cartesian gridpoints for x
            y_vals (np ndarray): cartesian gridpoints for y
            rho_vals (np ndarray): polar gridpoints for rho
            theta_vals (np ndarray): polar gridpoints for theta
            output_cart (bool): whether to also return the output in cartesian coordinates (keep polar form)
            skip_conv_calc (bool): whether to skip the convolution operator setup;
                intended for when the conv operators will be loaded (for interpolation to the cartesian grid)
            init_mode (str): choose which mode to use for initializing parameters
                Options:
                    [original, uniform-with-old-scale, normal-with-old-scale, he-normal]
        """
        super().__init__()

        self.N_h = N_h
        self.N_rho = N_rho
        self.c_1d = c_1d
        self.c_2d = c_2d
        self.w_1d = w_1d
        self.w_2d = w_2d
        self.N_cnn_1d = N_cnn_1d
        self.N_cnn_2d = N_cnn_2d

        init_mode = init_mode.lower() if init_mode is not None else "original"
        self.init_mode = init_mode

        # First, prepare interpolation operators
        self.output_as_cart = output_as_cart
        if output_as_cart:
            if x_vals is not None:
                # Assume that if x exists then all exist..
                self.x_vals = x_vals
                self.y_vals = y_vals if y_vals is not None else np.copy(x_vals)
                self.rho_vals = rho_vals
                self.theta_vals = theta_vals
            if not skip_conv_calc:
                self.prepare_polar_to_cart(interp_ready=False)

        # Set up neural networks parameters
        self.weight_dtype = torch.complex64
        self.real_dtype = torch.float32
        assert (
            self.w_2d % 2
        ), f"I can't figure out how to do padding for kernel sizes divisible by 2, {self.w_2d}"

        padding_1d = int(self.w_1d / 2 - 1) + 1
        padding_2d = int(self.w_2d / 2 - 1) + 1

        # 1. Initialize conv 1d parameters
        # in_channels_0 = self.N_h * 2
        # scale_0 = 2 / in_channels_0
        # params_0 = torch.nn.Parameter(
        #     scale_0
        #     * torch.rand(self.c_1d, in_channels_0, self.w_1d, dtype=self.weight_dtype)
        # )
        # self.conv_1d_layers = torch.nn.ParameterList([params_0])

        # for _ in range(self.N_cnn_1d - 2):
        #     scale_i = 2 / self.c_1d
        #     params_i = torch.nn.Parameter(
        #         scale_i
        #         * torch.rand(self.c_1d, self.c_1d, self.w_1d, dtype=self.weight_dtype)
        #     )
        #     self.conv_1d_layers.append(params_i)

        # scale_last = 2 / self.c_1d
        # params_last = torch.nn.Parameter(
        #     scale_last
        #     * torch.rand(self.N_rho, self.c_1d, self.w_1d, dtype=self.weight_dtype)
        # )
        # self.conv_1d_layers.append(params_last)

        ### Initialize parameter values ###
        ## 1D CNN
        # First layer
        # Dimensions along the freq/h axis
        self.c_1d_in  = self.N_h * 2
        self.c_1d_int = self.c_1d # internal layers
        self.c_1d_out = self.N_rho
        self.c_2d_in  = 1
        self.c_2d_int = self.c_2d
        self.c_2d_out = 1

        cnn1d_freq_h_axis_dims = [
            self.c_1d_in, # Input
            *((self.N_cnn_1d-1)*[self.c_1d_int]), # Interior
            self.c_1d_out,  # Final layer of 1D section
        ]
        logging.info(f"During the 1D Conv stage, freq/h axis has {cnn1d_freq_h_axis_dims} channels")

        self.conv_1d_layers = torch.nn.ParameterList([])
        for li in range(self.N_cnn_1d):
            h_dim_in  = cnn1d_freq_h_axis_dims[li]
            h_dim_out = cnn1d_freq_h_axis_dims[li+1]
            scale_old = 2 / h_dim_in
            scale_he  = torch.sqrt(torch.tensor(2 / h_dim_in, dtype=self.real_dtype))

            if init_mode == "original":
                scaling = scale_old
                rand_fn = torch.rand
            elif init_mode == "uniform-with-old-scale":
                scaling = scale_old
                rand_fn = lambda x: 2*torch.rand(x)-1
            elif init_mode == "normal-with-old-scale":
                scaling = scale_old
                rand_fn = torch.randn
            elif init_mode == "he-normal":
                rand_fn  = torch.randn
                scaling  = scale_he
            else:
                raise ValueError(
                    f"FYNet.__init__ received init_mode={init_mode} "
                    f"which is not in the recognized list of options: [original, "
                    f"uniform-with-old-scale, normal-with-old-scale, he-normal]",
                )
            new_params = scaling * rand_fn(
                h_dim_out,
                h_dim_in,
                self.w_1d,
                dtype=self.weight_dtype
            )

            self.conv_1d_layers.append(new_params)

        # 2. Initialize conv 2d parameters
        cnn2d_freq_h_axis_dims = [
            self.c_2d_in, # Input (=1)
            *((self.N_cnn_2d-1)*[self.c_2d_int]), # Interior
            self.c_2d_out, # Final layer (=1)
        ]
        logging.info(f"During the 2D Conv stage, freq/h axis has {cnn2d_freq_h_axis_dims} channels")
        self.conv_2d_layers = torch.nn.ParameterList([])
        for li in range(self.N_cnn_2d):
            freq_dim_in  = cnn2d_freq_h_axis_dims[li]
            freq_dim_out = cnn2d_freq_h_axis_dims[li+1]

            new_layer = torch.nn.Conv2d(
                in_channels=freq_dim_in,
                out_channels=freq_dim_out,
                kernel_size=self.w_2d,
                padding=padding_2d,
                padding_mode="zeros"
            )
            if init_mode == "he-normal":
                torch.nn.init.kaiming_normal_(new_layer.weight, nonlinearity="relu")
            self.conv_2d_layers.append(new_layer)

        self.relu = torch.nn.ReLU()

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor]:
        """Forward pass of the inversion model. Maps from wave fields to scattering
        objects. Here's a list of the intermediate shapes of the data
        (batch, N_M, N_H, 2) # Input shape
        -> (batch, N_M, N_H * 2) # Collapse the real and imag parts of the array into the H dimension
        -> (batch, N_H * 2, N_M) # Transpose to perform 1D convolution along the M dimension
        -> (batch, self.c_1d, N_M) # Channel dimension becomes whatever the specified self.c_1d is
        -> (batch, self.N_rho, N_M) # Last conv1d block outputs self.N_rho channels
        -> (batch, 1, self.N_rho, N_M) # Reshape to have a new channel dimension and do 2d conv on the final two dimesnions
        -> (batch, self.c_2d, self.N_rho, N_M) # During conv2d, the 2nd axis becomes the channel dimension
        -> (batch, self.N_rho, N_M) # Last conv2d block gets rid of the channel dimension
        -> (batch, N_M, self.N_rho) # Transpose to meet output shape requirements

        Args:
            x (torch.Tensor): Has shape (batch, N_M, N_H, 2)

        Returns:
            torch.Tensor: Has shape (batch, N_theta, N_rho); returns a tuple of tensors in case cartesian
            coordinates are requested
        """
        n_batch = x.shape[0]
        n_M = x.shape[1]

        # First, convolution across the N_M dimension.
        x = x.reshape((n_batch, n_M, -1))
        x = x.permute((0, 2, 1))

        for i, kernel_weights in enumerate(self.conv_1d_layers):
            x = conv_in_fourier_space(x, kernel_weights).real
            x = self.relu(x)

        # x has shape (n_batch, 2d_conv_channel_dim, 1d_conv_channel_dim, N_M)
        x = x.view((n_batch, 1, -1, n_M)) # (N_batch, 1, N_rho, N_m) where N_m=N_theta

        # Convolutions in the (N_rho, N_m) plane
        for i in range(self.N_cnn_2d):
            layer = self.conv_2d_layers[i]
            x = apply_conv_with_polar_padding(layer, x, angular_axis_last=True)

            if i == self.N_cnn_2d -1 :
                x = self.relu(x)


        out_polar = x.view((n_batch, -1, n_M)) # send to (N_batch, N_rho, N_theta)
        out_polar = out_polar.permute((0, 2, 1)) # send to (N_batch, N_theta, N_rho)
        result = out_polar

        if self.output_as_cart:
            out_cart = self.polar_to_cart(out_polar)
            result = out_polar, out_cart

        return result

    def __repr__(self) -> str:
        s = f"FYNetInverse model with {self.N_cnn_1d} 1D CNN layers, {self.N_cnn_2d} 2D CNN layers"
        s += f" 1D channel dim {self.c_1d}, 2D channel dim {self.c_2d}. 1D Convs performed with"
        s += f" {self.w_1d} modes, and 2D conv is performed with kernels of size ({self.w_2d}x{self.w_2d})."
        return s

    def to(self, device):
        """Override the base to() function to also move the interpolation operators"""
        # Referenced https://stackoverflow.com/questions/54706146/moving-member-tensors-with-module-to-in-pytorch
        moved_model = super(FYNetInverse, self).to(device)

        if "polar_to_x" in self.__dict__:
            moved_model.polar_to_x = self.polar_to_x.to(device)
            moved_model.polar_to_y = self.polar_to_y.to(device)
        return moved_model

    def prepare_polar_to_cart(self, interp_ready: bool = True):
        """Prepare the polar-to-cartesian interpolation operators"""
        center = np.zeros(2)  # could shift around the center later if we wanted
        assert "theta_vals" in self.__dir__()
        assert "rho_vals" in self.__dir__()

        N_rho = self.N_rho
        N_theta = self.theta_vals.shape[0]
        N_x = self.x_vals.shape[0]
        N_y = self.y_vals.shape[0]

        data_grid_xy = (
            np.array(np.meshgrid(self.x_vals, self.y_vals))
            .transpose(1, 2, 0)
            .reshape(N_x * N_y, 2)
        )
        cart_grid_radii = np.sqrt(data_grid_xy[:, 0] ** 2 + data_grid_xy[:, 1] ** 2)
        cart_grid_thetas = np.mod(
            np.arctan2(data_grid_xy[:, 1] - center[1], data_grid_xy[:, 0] - center[0]),
            2 * np.pi,
        )
        cart_grid_polar_coords = np.array([cart_grid_thetas, cart_grid_radii]).T

        padded_rho_vals, polar_padder = prep_polar_padder(
            self.rho_vals, N_theta, dim=1, with_torch=True
        )
        self.polar_padder = polar_padder

        if interp_ready:
            polar_to_x = self.polar_to_x
            polar_to_y = self.polar_to_y
        else:
            polar_to_x, polar_to_y = prep_conv_interp_2d(
                self.theta_vals,
                padded_rho_vals,
                cart_grid_polar_coords,
                bc_modes=("periodic", "extend"),
                a_neg_half=True,
            )
            # Convert the operations to pytorch formats
            self.polar_to_x = torch.sparse_csr_tensor(
                polar_to_x.indptr,
                polar_to_x.indices,
                polar_to_x.data,
                dtype=torch.float32,
                requires_grad=False,
            )
            self.polar_to_x = torch.tensor(
                polar_to_x.todense(), dtype=torch.float32, requires_grad=False
            )
            self.polar_to_y = torch.tensor(
                polar_to_y.todense(), dtype=torch.float32, requires_grad=False
            )

            logging.debug(f"Interp op shape info")
            logging.debug(f"Expected data shape: (n_b, {N_theta}, {N_rho})")
            logging.debug(f"polar_to_x shape: {polar_to_x.shape}")
            logging.debug(f"polar_to_y shape: {polar_to_y.shape}")
            logging.debug(f"Expected out shape: (n_b, {N_x}, {N_y})")

        def my_polar_to_cart(polar_data: np.ndarray) -> np.ndarray:
            """Helper function to send polar grid data to cartesian grid data
            Explicitly grabs values outside this function to reduce the chances of a weird scoping issue
            Note that this helper function also takes care of reshaping
            """
            nonlocal polar_padder, self
            nonlocal N_rho, N_theta, N_x, N_y
            # logging.debug("my_polar_to_cart: polar_data shape: %s", polar_data.shape)
            # logging.debug("my_polar_to_cart: polar_data type: %s", type(polar_data))
            res = polar_pad_and_apply(
                polar_padder,
                self.polar_to_x,
                self.polar_to_y,
                polar_data.reshape(-1, N_theta, N_rho),
                batched=True,
            ).reshape(
                -1, N_y, N_x
            )  # .transpose(1, 2)

            return res

        self.polar_to_cart = my_polar_to_cart
