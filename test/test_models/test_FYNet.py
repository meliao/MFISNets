import pytest
import torch
from src.models.FYNet import (
    FYNetForward,
    FYNetInverse,
)
from src.utils.test_utils import (
    check_arrays_close,
)

class Test_FYNetForward:
    def test_0(self: None) -> None:
        """Make sure it initializes and returns the correct shape"""

        N_M = 12
        N_H = 14
        N_theta = 12
        N_rho = 7
        batch = 5

        output_shape = (batch, N_M, N_H)
        input_shape = (batch, N_theta, N_rho)

        N_cnn_1d = 4
        c = 7
        w = 5

        x = FYNetForward(N_cnn_1d, c, w, N_rho, N_H)

        inputs = torch.randn(input_shape, dtype=torch.complex64)
        print("test: inputs shape: ", inputs.shape)
        y = x(inputs)

        assert y.shape == output_shape

    def test_1(self: None) -> None:
        """Test with 1 CNN layer to make sure dimensions are still correct."""

        N_M = 12
        N_H = 14
        N_theta = 12
        N_rho = 7
        batch = 5

        output_shape = (batch, N_M, N_H)
        input_shape = (batch, N_theta, N_rho)

        N_cnn_1d = 1
        c = 7
        w = 3

        x = FYNetForward(N_cnn_1d, c, w, N_rho, N_H)

        inputs = torch.randn(input_shape, dtype=torch.complex64)
        print("test: inputs shape: ", inputs.shape)
        y = x(inputs)

        assert y.shape == output_shape

        assert not torch.any(torch.isnan(y))

    @pytest.mark.skip()
    def test_2(self: None) -> None:
        """Tests that 1 CNN layer is the same as multiplying by a banded
        convolution matrix
        """
        N_batch = 1
        N_M = 1
        N_H = 12

        N_theta = 12
        N_rho = 1
        data = torch.zeros((N_batch, N_theta, N_rho))
        data[0, 0, 0] = 1.0
        data[0, 1, 0] = 3.0
        A = 1.0
        B = 2.0
        N = 12
        on_diag = A * torch.ones(N)
        off_diag = B * torch.ones(N - 1)

        banded_mat = (
            torch.diag(on_diag)
            + torch.diag(off_diag, diagonal=-1)
            + torch.diag(off_diag, diagonal=1)
        )
        banded_mat[0, N - 1] = B
        banded_mat[N - 1, 0] = B

        prod = torch.matmul(banded_mat, data)
        x = FYNetForward(1, 1, 3, 1, 1)
        conv_weights = torch.Tensor([[[B, A, B]]]).to(torch.complex64)
        assert conv_weights.shape == x.conv_1d_layers[0].weight.shape

        conv_weights = torch.nn.Parameter(conv_weights)
        # conv_weights = torch.nn.Parameter(conv_weights.view((2, 1, 3)))

        x.conv_1d_layers[0].weight = conv_weights
        x.conv_1d_layers[0].bias = torch.nn.Parameter(
            torch.zeros_like(x.conv_1d_layers[0].bias)
        )
        print("Weight dtype", x.conv_1d_layers[0].weight.dtype)
        print("Bias dtype", x.conv_1d_layers[0].bias.dtype)
        data = data.to(torch.complex64)
        out = x(data)
        print("Prod shape: ", prod.shape)
        print("Out shape: ", out.shape)
        check_arrays_close(prod.numpy(), out[:, :, :].real.detach().numpy())


class Test_FYNetInverse:
    def test_0(self: None) -> None:
        """Make sure it initializes and returns the correct shape"""

        N_M = 12
        N_H = 14
        N_theta = 12
        N_rho = 7
        batch = 5

        output_shape = (batch, N_theta, N_rho)
        input_shape = (batch, N_M, N_H, 2)

        N_cnn_1d = 4
        N_cnn_2d = 4
        c_1d = 5
        c_2d = 13
        w_1d = 5
        w_2d = 3

        x = FYNetInverse(N_H, N_rho, c_1d, c_2d, w_1d, w_2d, N_cnn_1d, N_cnn_2d)

        inputs = torch.randn(input_shape)
        y = x(inputs)

        assert y.shape == output_shape
        # assert False

    def test_1(self: None) -> None:
        """Make sure it initializes and returns the correct shape with 1 2d conv
        layer
        """

        N_M = 12
        N_H = 14
        N_theta = 12
        N_rho = 7
        batch = 5

        output_shape = (batch, N_theta, N_rho)
        input_shape = (batch, N_M, N_H, 2)

        N_cnn_1d = 4
        N_cnn_2d = 1
        c_1d = 7
        c_2d = 7
        w_1d = 5
        w_2d = 3

        x = FYNetInverse(N_H, N_rho, c_1d, c_2d, w_1d, w_2d, N_cnn_1d, N_cnn_2d)

        inputs = torch.randn(input_shape)
        y = x(inputs)

        assert y.shape == output_shape


if __name__ == "__main__":
    pytest.main()
