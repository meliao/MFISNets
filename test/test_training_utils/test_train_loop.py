import pytest
import torch
import numpy as np
from src.utils.test_utils import check_arrays_close

from src.training_utils.loss_functions import (
    psnr,
    _mse_along_batch,
)
from src.training_utils.train_loop import evaluate_losses_on_dataloader


class PredictAllOnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns an all-ones output same size as input"""
        return torch.ones_like(x)


class Test_evaluate_losses_on_dataloader:
    def test_0(self: None) -> None:
        loss_fn_dd = {
            "PSNR": lambda x, y, z: psnr(x, y),
            "MSE": lambda x, y, z: _mse_along_batch(x, y),
        }

        n_samples = 10
        batch_size = 3
        feature_dim = 3
        feature_dim_2 = 7
        X = torch.randn((n_samples, feature_dim, feature_dim_2))
        y = torch.zeros((n_samples, feature_dim, feature_dim_2))
        z = torch.zeros((n_samples, feature_dim, feature_dim_2))

        dset = torch.utils.data.TensorDataset(X, y, z)
        dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size)

        device = "cpu"

        model = PredictAllOnesModule()

        out_dd = evaluate_losses_on_dataloader(model, dloader, loss_fn_dd, device)

        assert out_dd["PSNR"].shape[0] == n_samples
        assert out_dd["MSE"].shape[0] == n_samples

        assert torch.all(out_dd["MSE"] != 0.0)

        check_arrays_close(out_dd["MSE"].numpy(), np.ones(n_samples))

    def test_1(self: None) -> None:
        loss_fn_dd = {
            "PSNR": lambda x, y, z: psnr(x, y),
            "MSE": lambda x, y, z: _mse_along_batch(x, y),
        }

        n_samples = 10
        batch_size = 3
        feature_dim = 3
        feature_dim_2 = 7
        X = torch.randn((n_samples, feature_dim, feature_dim_2))
        y = torch.randn((n_samples, feature_dim, feature_dim_2))
        z = torch.randn((n_samples, feature_dim, feature_dim_2))

        dset = torch.utils.data.TensorDataset(X, y, z)
        dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        model = PredictAllOnesModule()

        out_dd = evaluate_losses_on_dataloader(model, dloader, loss_fn_dd, device)

        expected_mse = _mse_along_batch(torch.ones_like(X), y)
        check_arrays_close(expected_mse.numpy(), out_dd["MSE"].numpy())

        expected_psnr = psnr(torch.ones_like(X), y)
        check_arrays_close(expected_psnr.numpy(), out_dd["PSNR"].numpy())

    def test_2(self: None) -> None:
        loss_fn_dd = {
            "PSNR": lambda x, y, z: psnr(x, y),
            "MSE": lambda x, y, z: _mse_along_batch(x, y),
        }

        n_samples = 27
        batch_size = 3
        feature_dim = 3
        feature_dim_2 = 7
        X = torch.randn((n_samples, feature_dim, feature_dim_2))
        y = torch.randn((n_samples, feature_dim, feature_dim_2))
        z = torch.randn((n_samples, feature_dim, feature_dim_2))

        dset = torch.utils.data.TensorDataset(X, y, z)
        dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        model = PredictAllOnesModule()

        out_dd = evaluate_losses_on_dataloader(model, dloader, loss_fn_dd, device)

        expected_mse = _mse_along_batch(torch.ones_like(X), y)
        check_arrays_close(expected_mse.numpy(), out_dd["MSE"].numpy())

        expected_psnr = psnr(torch.ones_like(X), y)
        check_arrays_close(expected_psnr.numpy(), out_dd["PSNR"].numpy())


if __name__ == "__main__":
    pytest.main()
