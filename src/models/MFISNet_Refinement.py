# MFISNet-Refinement
# Our proposed method, which progressively refines an estimate
# of the scattering potential given multi-frequency data.
#
# The method starts with low-frequency scattering measurements
# to create a low-spatial-frequency estimate of the scattering
# potential, and uses scattering data from successively higher
# frequencies to improve this estimate.
# This approach is loosely inspired by recursive linearization
# but does not follow the exact formulation due to the significant
# computational expense required.

import torch
from src.models.FYNet import FYNetInverse
from src.utils.conv_ops import apply_conv_with_polar_padding

from typing import Dict
import logging

class MFISNet_Refinement(torch.nn.Module):
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
        freq_pred_idx: int = None,
        return_all_q_hats: bool = False,
        init_mode: str = None,
    ) -> None:
        """
        We are calling this MFISNet-Refinement. It filters the
        concatenation of the intermeditate and final q_hats with extra 2D CNNs.

        Args:
            N_h (int): Input data shape -- number of gridpoints in the h axis
            N_rho (int): Output data shape -- number of gridpoitns in the radial axis.
            c_1d (int): Number of channels for 1d conv
            c_2d (int): Number of channels for 2d conv
            w_1d (int): Width of 1d conv kernel
            w_2d (int): Width of 2d conv kernel
            N_cnn_1d (int): Number of 1d conv layers
            N_cnn_2d (int): Number of 2d conv layers
            N_freqs (int): Number of frequencies in the input data. Controls the
                number of refinement blocks that are created.
            freq_pred_idx (int, optional): Specifies at which block the forward passes
                                            should return. Helpful for block-wise training.
                                            Defaults to None.
            return_all_q_hats (bool, optional): If True, all of the intermediate estimates of
                                                the scattering potential will be returned.
                                                Defaults to False.
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

        init_mode = init_mode.lower() if init_mode is not None else "original"
        self.init_mode = init_mode

        if freq_pred_idx is None:
            self.freq_pred_idx = self.N_freqs - 1
        else:
            self.freq_pred_idx = freq_pred_idx

        self.return_all_q_hats = return_all_q_hats

        self.fine_tune_bool = False

        self.inverse_networks = torch.nn.ModuleList()
        self.filtering_blocks = torch.nn.ModuleList()
        inv_model_0 = FYNetInverse(
            N_h=self.N_h,
            N_rho=self.N_rho,
            c_1d=self.c_1d,
            c_2d=self.c_2d,
            w_1d=self.w_1d,
            w_2d=self.w_2d,
            N_cnn_1d=self.N_cnn_1d,
            N_cnn_2d=self.N_cnn_2d,
            init_mode=init_mode,
        )
        self.inverse_networks.append(inv_model_0)
        for i in range(self.N_freqs - 1):
            inv_model = FYNetInverse(
                N_h=self.N_h,
                N_rho=self.N_rho,
                c_1d=self.c_1d,
                c_2d=self.c_2d,
                w_1d=self.w_1d,
                w_2d=self.w_2d,
                N_cnn_1d=self.N_cnn_1d,
                N_cnn_2d=self.N_cnn_2d,
                init_mode=init_mode,
            )
            self.inverse_networks.append(inv_model)

            filter_block = KLayer2DCNN(
                n_layers=self.N_cnn_2d,
                n_in_channels=2,
                n_out_channels=1,
                n_feature_channels=self.c_2d,
                kernel_size=self.w_2d,
                skip_connection=True,
                init_mode=init_mode,
            )
            self.filtering_blocks.append(filter_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): has shape (batch, N_freqs, N_m, N_h, 2 )

        Returns:
            torch.Tensor: has shape (batch, N_theta, N_rho), OR optionally
            (batch, N_freqs, N_theta, N_rho) if self.return_all_q_hats is True
        """
        q_hat = self.inverse_networks[0](x[:, 0])

        if self.freq_pred_idx == 0:
            return q_hat
        a, b, _ = q_hat.shape

        q_hat_out = torch.empty(
            (a, self.freq_pred_idx + 1, b, self.N_rho),
            dtype=torch.float32,
            device=x.device,
        )
        q_hat_out[:, 0] = q_hat

        for i in range(1, self.freq_pred_idx + 1):
            d_mh_in = x[:, i]

            q_hat_tilde = self.inverse_networks[i](d_mh_in).unsqueeze(1)
            other = q_hat_out[:, i - 1].unsqueeze(1)

            filtering_in = torch.cat((q_hat_tilde, other), dim=1)

            q_hat = self.filtering_blocks[i - 1](filtering_in).squeeze(1)

            q_hat_out[:, i] = q_hat

        if self.return_all_q_hats:
            return q_hat_out
        else:
            return q_hat_out[:, -1]

    def __repr__(self) -> str:
        s = f"MFISNet-Refinement object with {self.N_freqs} blocks, freq_pred_idx={self.freq_pred_idx}, return_all_q_hats={self.return_all_q_hats} "
        s += f"\n Here is the last block: {self.inverse_networks[-1]}"
        s += f"\n Here are whether the first parameter in each block is trainable: "
        for i, block in enumerate(self.inverse_networks):
            p = list(block.parameters())[0]
            s += f"block {i}: {p.requires_grad} "
        return s

# 2D CNN module that is used to combine outputs from different frequencies
class KLayer2DCNN(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_in_channels: int,
        n_out_channels: int,
        n_feature_channels: int,
        kernel_size: int,
        skip_connection: bool = False,
        init_mode: str = None,
        angular_axis_last: bool = False,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.n_feature_channels = n_feature_channels
        self.kernel_size = kernel_size
        self.skip_connection = skip_connection

        self.real_dtype = torch.float32

        init_mode = init_mode.lower() if init_mode is not None else "original"
        self.init_mode = init_mode

        padding_2d = int(kernel_size / 2 - 1) + 1
        self.cnn_layers = torch.nn.ParameterList()

        channel_dims = [
            n_in_channels,  # Input (=1)
            *((n_layers - 1) * [n_feature_channels]),  # Interior
            n_out_channels,  # Final layer (=1)
        ]
        logging.info(
            f"During the k-layer 2D CNN stage, the numbers of channels are {channel_dims}"
        )
        self.cnn_layers = torch.nn.ParameterList([])
        for li in range(n_layers):
            channel_dim_in = channel_dims[li]
            channel_dim_out = channel_dims[li + 1]

            new_layer = torch.nn.Conv2d(
                in_channels=channel_dim_in,
                out_channels=channel_dim_out,
                kernel_size=self.kernel_size,
                padding=padding_2d,
                padding_mode="zeros",
            )
            if init_mode == "he-normal":
                torch.nn.init.kaiming_normal_(new_layer.weight, nonlinearity="relu")
            self.cnn_layers.append(new_layer)

        self.relu = torch.nn.ReLU()
        self.angular_axis_last = angular_axis_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Assume x has shape (batch, self.n_in_channels, X, Y)
        Output will have shape (batch, self.n_out_channels, X, Y)

        If skip_connection is specified, then the slice (:, :-1, :, :)
        will be added to the output
        """
        if self.skip_connection:
            add_slice = x[:, -1].unsqueeze(1)

        for i, layer_i in enumerate(self.cnn_layers):
            x = apply_conv_with_polar_padding(
                layer_i, x, angular_axis_last=self.angular_axis_last
            )
            if i + 1 < self.n_layers:
                x = self.relu(x)
        out = x

        if self.skip_connection:
            out = out + add_slice

        return out


def load_MFISNet_Refinement_from_state_dict(
    state_dict: Dict,
) -> MFISNet_Refinement:
    """
    Given the state dict, this function figures out architecture hyperparameters,
    and then compiles a model, and then loads the weights.

    Args:
        state_dict (Dict): State dict generated by a torch nn Module's save() method

    Returns:
        MFISNet_Refinement: Model with loaded weights.
    """
    keys_lst = list(state_dict.keys())

    inverse_networks_keys = [k for k in keys_lst if "inverse_networks" in k]
    conv_1d_layers_keys = [
        k for k in keys_lst if "inverse_networks.0.conv_1d_layers" in k
    ]
    conv_2d_layers_keys = [
        k for k in keys_lst if "inverse_networks.0.conv_2d_layers" in k
    ]

    # Get the number of frequencies by the names of inverse_networks.*.conv_1d_layers
    # and then add 1
    layer_n_keys = [int(k.split(".")[1]) for k in inverse_networks_keys]
    N_freqs = max(layer_n_keys) + 1

    # print("Found N_freqs: %i" % N_freqs)

    layer_n_1d_cnns = [int(k.split(".")[3]) for k in conv_1d_layers_keys]
    N_cnn_1d = max(layer_n_1d_cnns) + 1
    layer_n_2d_cnns = [int(k.split(".")[3]) for k in conv_2d_layers_keys]
    N_cnn_2d = max(layer_n_2d_cnns) + 1

    weight_c_1d = state_dict["inverse_networks.0.conv_1d_layers.0"]

    c_1d, in_channels_0, w_1d = weight_c_1d.shape
    N_h = in_channels_0 // 2

    weight_c_2d = state_dict["inverse_networks.0.conv_2d_layers.0.weight"]
    # print("weight_c_2d shape: ", weight_c_2d.shape)
    c_2d, _, w_2d, _ = weight_c_2d.shape

    model = MFISNet_Refinement(
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
