from typing import Dict
import pytest
import torch

from src.models.MFISNet_Parallel import (
    MFISNet_Parallel,
    load_MFISNet_Parallel_from_state_dict,
)

class Test_MFISNet_Parallel:
    def test_0(self) -> None:
        """Makes sure everything initializes correctly,
        forward pass without error, and returns the correct shape.
        """
        N_h = 7
        N_rho = 7
        c_1d = 4
        c_2d = 4
        w_1d = 3
        w_2d = 3
        N_cnn_1d = 3
        N_cnn_2d = 3
        N_freqs = 3

        n_batch = 5
        N_m = 12

        model = MFISNet_Parallel(
            N_h=N_h,
            N_rho=N_rho,
            c_1d=c_1d,
            c_2d=c_2d,
            w_1d=w_1d,
            w_2d=w_2d,
            N_cnn_1d=N_cnn_1d,
            N_cnn_2d=N_cnn_2d,
            N_freqs=N_freqs,
        )

        x_in = torch.randn((n_batch, N_freqs, N_m, N_h, 2))
        x_out = model(x_in)
        assert x_out.shape == (n_batch, N_m, N_rho)


@pytest.fixture
def model_params_1() -> Dict:
    """One setting of model params
    for ParallelFYNetInverse
    """
    return {
        "N_h": 10,
        "N_rho": 10,
        "c_1d": 4,
        "c_2d": 5,
        "w_1d": 3,
        "w_2d": 7,
        "N_cnn_1d": 3,
        "N_cnn_2d": 3,
        "N_freqs": 3,
    }

@pytest.fixture
def state_dict_1(model_params_1: Dict) -> Dict:
    """Creates a state dict for
    the model with the parameters model_params_1
    """
    model = MFISNet_Parallel(**model_params_1)
    return model.state_dict()


class Test_load_ResidualFYNetInverseAAA_from_state_dict:
    def test_0(self, model_params_1: Dict, state_dict_1: Dict) -> None:
        """Makes sure the model compiles and runs without error."""
        model = load_MFISNet_Parallel_from_state_dict(state_dict_1)

        assert model.N_h == model_params_1["N_h"]
        assert model.N_rho == model_params_1["N_rho"]
        assert model.c_1d == model_params_1["c_1d"]
        assert model.c_2d == model_params_1["c_2d"]
        assert model.w_1d == model_params_1["w_1d"]
        assert model.w_2d == model_params_1["w_2d"]
        assert model.N_cnn_1d == model_params_1["N_cnn_1d"]
        assert model.N_cnn_2d == model_params_1["N_cnn_2d"]
        assert model.N_freqs == model_params_1["N_freqs"]


if __name__ == "__main__":
    pytest.main()
