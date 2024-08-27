import pytest
import torch
from typing import Dict
from src.models.MFISNet_Refinement import (
    MFISNet_Refinement,
    KLayer2DCNN,
    load_MFISNet_Refinement_from_state_dict,
)

TMP_DIR = "tmp/"

class Test_MFISNet_Refinement:
    def test_0(self) -> None:
        """Makes sure everything initializes correctly,
        forward pass without error, and returns the correct shape. When input_q_hat=False
        """
        N_h = 10
        N_rho = 10
        c_1d = 4
        c_2d = 4
        w_1d = 3
        w_2d = 3
        N_cnn_1d = 3
        N_cnn_2d = 3
        N_freqs = 3
        n_batch = 5
        N_m = 12

        model = MFISNet_Refinement(
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

        for i in range(N_freqs):
            model.freq_pred_idx = i
            print("Looking at freq_pred_idx: %i" % i)
            x_in = torch.randn((n_batch, N_freqs, N_m, N_h, 2))

            x_out = model(x_in)

            assert x_out.shape == (n_batch, N_m, N_rho)

    def test_1(self) -> None:
        """Same as test_0, but here input_q_hat = True"""
        N_h = 10
        N_rho = 10
        c_1d = 4
        c_2d = 4
        w_1d = 3
        w_2d = 3
        N_cnn_1d = 3
        N_cnn_2d = 3
        N_freqs = 3

        n_batch = 5
        N_m = 12

        model = MFISNet_Refinement(
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

    def test_2(self) -> None:
        """Same as test_1, but here freq_pred_idx = 0"""
        N_h = 10
        N_rho = 10
        c_1d = 4
        c_2d = 4
        w_1d = 3
        w_2d = 3
        N_cnn_1d = 3
        N_cnn_2d = 3
        N_freqs = 3

        n_batch = 5
        N_m = 12

        model = MFISNet_Refinement(
            N_h=N_h,
            N_rho=N_rho,
            c_1d=c_1d,
            c_2d=c_2d,
            w_1d=w_1d,
            w_2d=w_2d,
            N_cnn_1d=N_cnn_1d,
            N_cnn_2d=N_cnn_2d,
            N_freqs=N_freqs,
            freq_pred_idx=0,
        )

        x_in = torch.randn((n_batch, N_freqs, N_m, N_h, 2))

        x_out = model(x_in)

        assert x_out.shape == (n_batch, N_m, N_rho)


class Test_KLayer2DCNN:
    def test_0(self) -> None:
        """Makes sure the model compiles and runs without error."""
        N_x = 10
        N_y = 10
        c_in = 4
        c_out = 5
        c_feature = 7
        w = 3
        N_cnn = 3
        batch = 5

        model = KLayer2DCNN(
            n_layers=N_cnn,
            n_in_channels=c_in,
            n_out_channels=c_out,
            n_feature_channels=c_feature,
            kernel_size=w,
        )

        x_in = torch.randn((batch, c_in, N_x, N_y))

        x_out = model(x_in)

        assert x_out.shape == (batch, c_out, N_x, N_y)

    def test_0(self) -> None:
        """Makes sure the model compiles and runs without error."""
        N_x = 10
        N_y = 10
        c_in = 4
        c_out = 5
        c_feature = 7
        w = 3
        N_cnn = 3
        batch = 5

        model = KLayer2DCNN(
            n_layers=N_cnn,
            n_in_channels=c_in,
            n_out_channels=c_out,
            n_feature_channels=c_feature,
            kernel_size=w,
            skip_connection=True,
        )

        x_in = torch.randn((batch, c_in, N_x, N_y))

        x_out = model(x_in)

        assert x_out.shape == (batch, c_out, N_x, N_y)


@pytest.fixture
def model_params_1() -> Dict:
    """One setting of model params
    for MFISNetRefinement
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
def state_dict_refinement(model_params_1: Dict) -> Dict:
    """Creates a state dict for
    the model with the parameters model_params_1
    """
    model = MFISNet_Refinement(**model_params_1)
    return model.state_dict()


class Test_load_MFISNet_Refinement_from_state_dict:
    def test_0(self, model_params_1: Dict, state_dict_refinement: Dict) -> None:
        """Makes sure the model compiles and runs without error."""
        model = load_MFISNet_Refinement_from_state_dict(state_dict_refinement)

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
