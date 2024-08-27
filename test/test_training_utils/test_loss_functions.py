import pytest
import torch
import numpy as np
from src.utils.test_utils import check_scalars_close

from src.training_utils.loss_functions import (
    psnr,
    relative_l2_error,
)

class Test_psnr:
    def test_0(self: None) -> None:
        """Just make sure the thing returns the correct shape.
        """
        shape = (5, 7, 3)
        a = torch.randn(shape)
        b = torch.randn(shape)

        o = psnr(a, b)
        assert o.shape == (5,)

    def test_1(self: None) -> None:

        a = torch.tensor([[2., 2., 3.]])
        b = torch.tensor([[1., 2., 3.]])

        mse = 1.0 / 3.0
        max_sq = 4.0
        expected_out = 10 * np.log10(max_sq / mse)

        out = psnr(a, b)

        check_scalars_close(expected_out, out.item())

class Test_relative_l2_error:
    def test_0(self: None) -> None:
        """
        Just make sure things return correct shape
        """
        shape = (5, 7, 3)
        a = torch.randn(shape)
        b = torch.randn(shape)

        o = relative_l2_error(a, b)
        assert o.shape == (5,)

    def test_1(self: None) -> None:
        a = torch.tensor([[1. + 0.j, 0. + 0.j]])
        b = torch.tensor([[1. + 0.j, 1. + 0.j]])

        nrm_a = 1
        nrm_diffs = 1.
        o = relative_l2_error(b, a)
        check_scalars_close(o.item(), nrm_diffs / nrm_a)


    def test_2(self: None) -> None:

        a = torch.tensor([[np.sqrt(2), np.sqrt(2)]])
        a_nrm = 2.
        b = torch.tensor([[1., 1.]])
        b_nrm = np.sqrt(2)
        a_minus_b_nrm = np.sqrt((1 - np.sqrt(2)) ** 2 + (1 - np.sqrt(2)) ** 2)

        o = relative_l2_error(a, b)
        check_scalars_close(o.item(), a_minus_b_nrm / b_nrm)

        o2 = relative_l2_error(b, a)
        check_scalars_close(o2.item(), a_minus_b_nrm / a_nrm)

if __name__ == "__main__":
    pytest.main()