# MFISNet-Parallel
# Inspired by FYNet but uses different adjoint networks
# for each input frequency, each runs in parallel.
# The outputs are merged for the 2d conv filtering step.

import torch
import numpy as np
from src.utils.conv_ops import (
    conv_in_fourier_space,
    apply_conv_with_polar_padding,
)

from typing import Tuple, Dict


class MFISNet_Parallel(torch.nn.Module):
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
        N_freqs: int,
    ) -> None:
        """The inverse NN described in section 2 of FY19. NN inputs have shape
        (batch, N_M, N_H, 2) and outputs have shape (batch, N_theta, N_rho),
        and we assume N_M == N_theta.

        Args:
            c_1d (int): Number of channels for 1d conv
            c_2d (int): Number of channels for 2d conv
            w_1d (int): Width of 1d conv kernel
            w_2d (int): Width of 2d conv kernel
            N_cnn_1d (int): Number of 1d conv layers
            N_cnn_2d (int): Number of 2d conv layers

            x_vals (np ndarray): cartesian gridpoints for x
            y_vals (np ndarray): cartesian gridpoints for y
            rho_vals (np ndarray): polar gridpoints for rho
            theta_vals (np ndarray): polar gridpoints for theta
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
        self.N_freqs = N_freqs

        # Set up neural networks parameters
        self.weight_dtype = torch.complex64
        assert (
            self.w_2d % 2
        ), f"I can't figure out how to do padding for kernel sizes divisible by 2, {self.w_2d}"

        padding_1d = int(self.w_1d / 2 - 1) + 1
        padding_2d = int(self.w_2d / 2 - 1) + 1

        # 1. Initialize conv 1d parameters in a ParameterList saved at self.conv_1d_layers.
        # This list contains all of the parameters for all of the 1D conv heads, cycling
        # through the heads. So it will be <N_freqs> copies of the first layer, then
        # <N_freqs> copies of the second layer, and so on.
        in_channels_0 = self.N_h * 2
        scale_0 = 2 / in_channels_0  # new

        self.conv_1d_layers = torch.nn.ParameterList([])
        for _ in range(self.N_freqs):
            params_0 = torch.nn.Parameter(
                scale_0
                * torch.rand(
                    self.c_1d, in_channels_0, self.w_1d, dtype=self.weight_dtype
                )
            )
            self.conv_1d_layers.append(params_0)

        # Range over the conv layers that are not the first or last layers.
        for _ in range(self.N_cnn_1d - 2):
            scale_i = 2 / self.c_1d

            # Range over the different frequency heads.
            for _ in range(self.N_freqs):
                params_i = torch.nn.Parameter(
                    scale_i
                    * torch.rand(
                        self.c_1d, self.c_1d, self.w_1d, dtype=self.weight_dtype
                    )
                )
                self.conv_1d_layers.append(params_i)

        scale_last = 2 / self.c_1d

        for _ in range(self.N_freqs):
            params_last = torch.nn.Parameter(
                scale_last
                * torch.rand(self.N_rho, self.c_1d, self.w_1d, dtype=self.weight_dtype)
            )
            self.conv_1d_layers.append(params_last)

        # 2. Initialize conv 2d parameters
        if self.N_cnn_2d > 1:
            self.conv_2d_layers = torch.nn.ParameterList(
                [
                    torch.nn.Conv2d(
                        in_channels=self.N_freqs,
                        out_channels=self.c_2d,
                        kernel_size=self.w_2d,
                        padding=padding_2d,
                        padding_mode="circular",
                    )
                ]
            )

            for _ in range(self.N_cnn_2d - 2):
                self.conv_2d_layers.append(
                    torch.nn.Conv2d(
                        in_channels=self.c_2d,
                        out_channels=self.c_2d,
                        kernel_size=self.w_2d,
                        padding=padding_2d,
                        padding_mode="circular",
                    )
                )
            # Append the last layer that outputs N_rho channels
            self.conv_2d_layers.append(
                torch.nn.Conv2d(
                    in_channels=self.c_2d,
                    out_channels=1,
                    kernel_size=self.w_2d,
                    padding=padding_2d,
                    padding_mode="circular",
                )
            )
        else:
            self.conv_2d_layers = torch.nn.ParameterList(
                [
                    torch.nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=self.w_2d,
                        padding=padding_2d,
                        padding_mode="circular",
                    )
                ]
            )

        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor]:
        """Forward pass of the inversion model. Maps from wave fields to scattering
        objects. Here's a list of the intermediate shapes of the data
        (batch, N_freqs, N_M, N_H, 2) # Input shape
        -> (batch, N_freqs N_M, N_H * 2) # Collapse the real and imag parts of the array into the H dimension
        -> (batch, N_freqs, N_H * 2, N_M) # Transpose to perform 1D convolution along the M dimension
        -> (batch, N_freqs, self.c_1d, N_M) # Channel dimension becomes whatever the specified self.c_1d is
        -> (batch, N_freqs, self.N_rho, N_M) # Last conv1d block outputs self.N_rho channels
        -> (batch, N_freqs, self.N_rho, N_M) # In the original FYNetInverse, this is where we would
                                            reshape to have a new channel dimension and do 2d conv on the final two dimesnions
        -> (batch, self.c_2d, self.N_rho, N_M) # During conv2d, the 2nd axis becomes the channel dimension
        -> (batch, self.N_rho, N_M) # Last conv2d block gets rid of the channel dimension
        -> (batch, N_M, self.N_rho) # Transpose to meet output shape requirements

        Args:
            x (torch.Tensor): Has shape (batch, N_freqs, N_M, N_H, 2)

        Returns:
            torch.Tensor: Has shape (batch, N_theta, N_rho)
        """
        n_batch, N_freqs, n_M, _, _ = x.shape
        assert N_freqs == self.N_freqs

        # First, reshape the imag and real channels into one dimension.
        x = x.reshape((n_batch, self.N_freqs, n_M, -1))
        # Next, put the M dimension last.
        x = x.permute((0, 1, 3, 2))

        x_lst = [x[:, i] for i in range(self.N_freqs)]
        for i in range(self.N_cnn_1d):
            for j in range(self.N_freqs):
                weights_i = self.conv_1d_layers[j + i * self.N_freqs]
                x_i = x_lst[j]
                x_i = conv_in_fourier_space(x_i, weights_i).real
                x_lst[j] = self.relu(x_i)

        x_lst_unsqueezed = [z.unsqueeze(1) for z in x_lst]
        x = torch.cat(x_lst_unsqueezed, dim=1)

        for i in range(self.N_cnn_2d - 1):
            layer = self.conv_2d_layers[i]
            x = apply_conv_with_polar_padding(layer, x)
            x = self.relu(x)

        last_layer = self.conv_2d_layers[-1]
        x = apply_conv_with_polar_padding(last_layer, x)

        out_polar = x.view((n_batch, -1, n_M))
        out_polar = out_polar.permute((0, 2, 1))
        result = out_polar

        return result


def load_MFISNet_Parallel_from_state_dict(state_dict: Dict) -> MFISNet_Parallel:
    """
    Load a MFISNet_Parallel model from a state dict.

    Args:
        state_dict (Dict): State dict generated by a torch nn Module's save() method

    Returns:
        MFISNet_Parallel: Model with loaded weights
    """
    print(
        "load_MFISNet_Parallel_from_state_dict: state_dict keys:",
        state_dict.keys(),
    )
    keys_lst = list(state_dict.keys())

    conv_1d_layers_keys = [k for k in keys_lst if "conv_1d_layers" in k]
    conv_2d_layers_keys = [k for k in keys_lst if "conv_2d_layers" in k]

    layer_n_1d_cnns = [int(k.split(".")[1]) for k in conv_1d_layers_keys]
    N_1d_layers = max(layer_n_1d_cnns) + 1
    layer_n_2d_cnns = [int(k.split(".")[1]) for k in conv_2d_layers_keys]
    N_cnn_2d = max(layer_n_2d_cnns) + 1

    weight_c_1d = state_dict["conv_1d_layers.0"]
    c_1d, in_channels_0, w_1d = weight_c_1d.shape
    N_h = in_channels_0 // 2

    weight_c_2d = state_dict["conv_2d_layers.0.weight"]
    c_2d, N_freqs, w_2d, _ = weight_c_2d.shape
    N_cnn_1d = N_1d_layers // N_freqs

    model = MFISNet_Parallel(
        N_h=N_h,
        N_rho=N_h,
        c_1d=c_1d,
        c_2d=c_2d,
        w_1d=w_1d,
        w_2d=w_2d,
        N_cnn_1d=N_cnn_1d,
        N_cnn_2d=N_cnn_2d,
        N_freqs=N_freqs,
    )
    model.load_state_dict(state_dict)
    return model
