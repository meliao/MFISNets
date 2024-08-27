from typing import Dict, Tuple
import pytest
import torch
from torch import nn
from src.utils.conv_ops import (
    polar_conv_padder,
    apply_conv_with_polar_padding,
    conv_in_fourier_space,
)
from src.utils.test_utils import (
    check_arrays_close,
)

class Test_polar_conv:
    def get_random_tensor(self, tensor_shape: Tuple[int], rng_seed=0) -> torch.Tensor:
        # Save and set the seed
        prev_seed = torch.seed()
        torch.manual_seed(rng_seed)
        rand_tensor = torch.normal(0, 1, tensor_shape)

        # Reset the rng state
        torch.manual_seed(prev_seed)
        return rand_tensor

    def get_random_filter(self, num_channels_out: int, num_channels_in: int, w_2d: int, rng_seed=2) -> torch.Tensor:
        # Save and set the seed
        prev_seed = torch.seed()
        torch.manual_seed(rng_seed)
        rand_filter_2d = torch.normal(0, 1, (num_channels_out, num_channels_in, w_2d, w_2d))
        # Reset the rng state
        torch.manual_seed(prev_seed)
        return rand_filter_2d

    def test_polar_padding_rotation(self) -> None:
        # Set up the tensor
        N_rho   = 40
        N_theta = 80
        num_channels_in  = 1
        num_channels_out = 5
        rand_tensor = self.get_random_tensor((num_channels_in, N_rho, N_theta), rng_seed=2345)

        # Set up the filter and conv2d layer
        w_2d = 7
        pad_width = (int(w_2d / 2 - 1) + 1)
        rand_filter = self.get_random_filter(num_channels_out, num_channels_in, w_2d, rng_seed=823094)
        conv2d_layer = nn.Conv2d(
            num_channels_in,
            num_channels_out,
            kernel_size=w_2d,
            bias=False,
            padding="same",
        )
        with torch.no_grad():
            conv2d_layer.weight.set_(rand_filter)

        # Compare the usual evaluation against a rolled version (equivariant to rotations about the origin)
        # center_slice = np.s_[..., pad_width: -pad_width, pad_width: -pad_width]
        dupl_rho_zero = True
        output_tensor = apply_conv_with_polar_padding(conv2d_layer, rand_tensor, pad_width, duplicate_rho_zero=dupl_rho_zero)

        # Rolled version (roll along the angular axis (the last one)
        roll_amount = 8
        tensor_rolled = torch.roll(rand_tensor, roll_amount, dims=(-1,))
        output_tensor_rolled = apply_conv_with_polar_padding(conv2d_layer, tensor_rolled, pad_width)
        output_tensor_alt = torch.roll(output_tensor_rolled, -roll_amount, dims=(-1,))

        sqrel_err_num   = torch.sum(torch.abs(output_tensor-output_tensor_alt)**2)
        sqrel_err_denom = torch.sum(torch.abs(output_tensor)**2)
        rel_err = float(torch.sqrt( sqrel_err_num / sqrel_err_denom ).item())
        print(f"Relative error: {rel_err:.3e}")
        assert rel_err < 1e-5

    def test_polar_padding_inspect_values(self) -> None:
        # Set up the tensor
        N_rho   = 40
        N_theta = 80
        num_channels_in  = 1
        num_channels_out = 5
        rand_tensor = self.get_random_tensor((num_channels_in, N_rho, N_theta), rng_seed=29345)

        # Set up the filter and conv2d layer
        w_2d = 7
        pad_width = (int(w_2d / 2 - 1) + 1)
        # Compare the usual evaluation against a rolled version (equivariant to rotations about the origin)
        # center_slice = np.s_[..., pad_width: -pad_width, pad_width: -pad_width]
        dupl_rho_zero = True
        padded_tensor, _center_slice = polar_conv_padder(rand_tensor, w_2d, pad_width=None, duplicate_rho_zero=dupl_rho_zero)

        # Verify that the interior nodes match
        # At the moment the padding function duplicates the rho=0 entries
        # This was not intended but used in all the experiments.
        has_dupl_rho_zero = dupl_rho_zero # True

        assert (N_theta % 2) == 0 # the test below only works for an even number of angular grid points
        interior_vals = rand_tensor[..., 1: pad_width,  :] # take at the smallest non-zero radial grid point
        radial_offset = -1 if has_dupl_rho_zero else -0 # offset from the center slice
        padded_vals   = padded_tensor[..., 1+radial_offset: pad_width+radial_offset, pad_width:-pad_width]
        padded_roll_angle = rand_tensor.shape[-1] // 2
        padded_vals_rolled = torch.roll(padded_vals, padded_roll_angle, dims=-1)
        # The padded values should match their source now:
        padded_vals_matched = torch.flip(padded_vals_rolled, dims=(-2,))
        padding_err = torch.linalg.norm((padded_vals_matched - interior_vals).flatten()) \
            / torch.linalg.norm(interior_vals.flatten())

        print(f"Relative padding error: {padding_err:.3e}")
        assert padding_err < 1e-5


class Test_conv_in_fourier_space:
    def test_0(self: None) -> None:
        """Manually-designed kernel and signal"""
        kernel = torch.zeros((1, 5), dtype=torch.complex64)
        kernel[0, 0] = 1.0
        signal_fourier = torch.zeros((1, 1, 5), dtype=torch.complex64)
        signal_fourier[0, 0, 0] = 1.0

        signal = torch.fft.ifft(torch.fft.ifftshift(signal_fourier))

        kernel = torch.zeros((1, 1, 5), dtype=torch.complex64)
        kernel[0, 0, 0] = 1.0
        out = conv_in_fourier_space(signal, kernel)

        check_arrays_close(signal.numpy(), out.numpy())

        # Create signal_2 by adding an orthogonal component in Fourier space.
        # This should not change the output because the kernel does not have
        # any weight there.
        signal_fourier_2 = signal_fourier
        signal_fourier_2[0, 0, 1] = 1.0
        signal_2 = torch.fft.ifft(torch.fft.ifftshift(signal_fourier_2))

        out = conv_in_fourier_space(signal_2, kernel)

        check_arrays_close(signal.numpy(), out.numpy())

    def test_1(self: None) -> None:
        """Just checks the sizes are correct"""
        batch = 3
        n_channels_in = 5
        signal_len = 35

        n_channels_out = 4
        kernel_len = 13

        signal_shape = (batch, n_channels_in, signal_len)
        kernel_shape = (n_channels_out, n_channels_in, kernel_len)
        output_shape = (batch, n_channels_out, signal_len)

        signal = torch.randn(signal_shape) + 1j * torch.randn(signal_shape)

        kernel = torch.randn(kernel_shape) + 1j * torch.randn(kernel_shape)

        out = conv_in_fourier_space(signal, kernel)

        assert out.shape == output_shape


if __name__ == "__main__":
    pytest.main()
