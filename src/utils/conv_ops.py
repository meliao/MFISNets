# Convolution operations for use by the different neural networks
# Includes:
# - 1d convolution in fourier space
# - padding for a polar grid (for the boundary conditions to make sense)
# - applying a 2d conv operation to the polar grid (using the padding)

import torch
import numpy as np


def conv_in_fourier_space(signal: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Perform a 1d convolution in Fourier space (along the last axis).
    Intended for kernels with broad spatial support.
    Assumes kernel is already in Fourier space

    Args:
        signal (torch.Tensor): Has shape (batch, N_channels_in, signal_len)
        kernel (torch.Tensor): has shape (N_channels_out, N_channels_in, kernel_len)

    Returns:
        torch.Tensor: Has shape (batch, N_channels_out, signal_len)
    """
    signal_fourier = torch.fft.fft(signal, signal.size(-1))
    signal_fourier = torch.fft.fftshift(signal_fourier)

    # Step 2: Zero-pad the kernel to match the length of the signal
    # Assuming kernel_len is an odd number, we can do zero-padding to make the kernel and signal have the same length.
    pad_left = (signal.size(-1) - kernel.size(-1)) // 2
    pad_right = signal.size(-1) - kernel.size(-1) - pad_left

    # Shape (N_channels_out, N_channels_in signal_len)
    kernel_padded = torch.nn.functional.pad(
        kernel, (pad_left, pad_right), "constant", 0
    )

    # Step 3: Perform pointwise multiplication with the kernel
    # Each batch element is shaped (N_channels_in, signal_len) and the kernel is
    # (N_channels_out, n_channels_in, signal_len), so for each output channel
    # of the kernel, this is a dot product between 2D arrays
    output_fourier = torch.einsum("abc,dbc->adc", signal_fourier, kernel_padded)

    # Step 4: Perform the inverse Fourier transform to get the final output
    output_fourier = torch.fft.ifftshift(output_fourier)
    output = torch.fft.ifft(output_fourier, signal.size(-1))

    return output


def apply_conv_with_polar_padding(
    conv2d_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    pad_width: int = None,
    duplicate_rho_zero: bool = True,
) -> torch.Tensor:
    """
    Helper function that applies the conv2d operation on a polar grid
    and uses the polar_padder function to get the correct boundary behavior.

    Assumes the radial and angular axes are the second-to-last and last axes, respectively
    e.g., (N_batch, N_channels, N_rho, N_theta) would be a possible shape
    """
    k1, k2 = conv2d_layer.kernel_size
    w_2d = max(k1, k2)  # should be square but in case it's not..
    pad_width = pad_width if pad_width is not None else (int(w_2d / 2 - 1) + 1)

    # 1-3. Perform the padding for the polar grid
    padded_tensor, center_slice = polar_conv_padder(
        input_tensor, w_2d, pad_width, duplicate_rho_zero=duplicate_rho_zero
    )

    # 4. Run the actual convolution layer then crop out the padding
    output_tensor = conv2d_layer(padded_tensor)[center_slice]

    return output_tensor


def polar_conv_padder(
    input_tensor: torch.Tensor,
    w_2d: int,
    pad_width: int = None,
    duplicate_rho_zero: bool = True,
) -> torch.Tensor:
    """Helper function that performs the padding to reflect the proper boundary conditions for Conv2D on a polar grid

    Assumes the radial and angular axes are the second-to-last and last axes, respectively
    e.g., (N_batch, N_channels, N_rho, N_theta) would be a possible shape

    Args:
        input_tensor (torch Tensor): un-padded tensor in polar coordinates to be padded.
            This tensor is expected to have shape (..., N_rho, N_theta)
        w_2d (int): width of the (square) 2d kernel
        pad_width (int): how much to pad on each side of both the radial and angular axes
        duplicate_rho_zero (bool): choose whether to duplicate the entries corresponding to rho=0
            This option is provided because all the experiments were run with the entries duplicated
            although it was not intentional.
    Returns:
        padded_tensor (torch Tensor): the resulting padded tensor
            with shape(..., N_rho+2*pad_width, N_theta+2*pad_width)
        center_slice (np slice object): a slice representing the original region
    """
    # Hard-coded to simplify the indexing logic
    radial_axis = -2
    angular_axis = -1

    pad_width = pad_width if pad_width is not None else (int(w_2d / 2 - 1) + 1)

    # Prepare the shapes
    input_shape = input_tensor.shape
    N_theta = input_shape[angular_axis]
    padded_shape = [*input_shape]
    padded_shape[radial_axis] += 2 * pad_width
    padded_shape[angular_axis] += 2 * pad_width
    padded_shape = tuple(padded_shape)
    padded_tensor = torch.zeros(padded_shape, device=input_tensor.device)

    # 1. Start by copying over the original values into the center
    center_slice = np.s_[..., pad_width:-pad_width, pad_width:-pad_width]
    padded_tensor[center_slice] = input_tensor

    # 2. Extend in the radial direction
    # fetches the padding on the radial side close to the origin
    # The exterior is assumed to be zero-padded and therefore does not
    # need to be touched
    radial_start = 0 if duplicate_rho_zero else 1
    values_to_fill_in = torch.flip(
        input_tensor[..., radial_start : radial_start + pad_width, :],
        dims=(radial_axis,),
    )
    values_to_fill_in = torch.roll(values_to_fill_in, shifts=N_theta // 2, dims=-1)

    # 2.5. Make adjustments for odd-numbered outputs
    if N_theta % 2 == 1:
        # CAUTION: NOT FULLY TESTED (as of 5/28/24)
        # (the main concern is whether there's an off-by-one indexing error)
        raise NotImplementedError(
            f"Code is untested for N_theta={N_theta} odd; "
            f"use with caution (this message can be commented out)"
        )
        cubic_interp_filter = np.array([-1.0, 9.0, 9.0, -1.0]) / 16
        values_to_fill_in = circular_convolve1d(
            values_to_fill_in, cubic_interp_filter, roll_offset=2, axis=angular_axis
        )
    padded_tensor[..., :pad_width, pad_width:-pad_width] = values_to_fill_in

    # 3. Extend in the angular direction
    # Left side
    padded_tensor[..., :, :pad_width] = padded_tensor[
        ..., :, -2 * pad_width : -pad_width
    ]
    # Right side
    padded_tensor[..., :, -pad_width:] = padded_tensor[
        ..., :, pad_width : 2 * pad_width
    ]
    return padded_tensor, center_slice


def circular_convolve1d(
    input_tensor: torch.Tensor,
    filter_1d: torch.Tensor,
    roll_offset: int = 0,
    axis: int = -1,
) -> torch.Tensor:
    """Helper function that performs a simple convolution as a series of roll-and-adds
    Intended to perform interpolation to pad polar grids with an odd number of angular gridpoints.

    Intended for very small filter_1d, for example for cubic interpolation
    (for input points at (-1.5, -0.5, 0.5, 1.5) the weights to calculate the
    value at 0 are given by [-1/16, 9/16, 9/16, -1/16]); this function uses
    a shift-and-add approach which is quite inefficient for long filters.
    (this is a helper function since pytorch's conv1d function had more shape restrictions)

    Args:
        input_tensor (torch Tensor): input tensor to be convolved against
        filter_1d    (torch Tensor): one-dimensional filter (broadcasted to all other axes)
        roll_offset (int): specify how much to rotate/roll the input based on the filter
            This is a bit of a fudge factor to make it easier to fix index errors
        axis (int): specify which axis the filter should be applied to
    Returns:
        output_tensor (torch Tensor): the result of the cicrular convolution

    """
    dims_arg = axis if isinstance(axis, tuple) else (axis,)
    output_tensor = filter_1d[0] * (
        input_tensor
        if roll_offset == 0
        else torch.roll(input_tensor, roll_offset, dims=dims_arg)
    )
    for i in range(1, len(filter_1d)):
        output_tensor += filter_1d[i] * torch.roll(
            input_tensor, roll_offset - i, dims=dims_arg
        )
    return output_tensor
