import pytest
import numpy as np
import os, shutil, time
import torch

from train_MFISNet_Fused import main as train_mfisnet_fused_main, load_multifreq_dataset

from src.data.data_naming_constants import (
    Q_CART,
    Q_POLAR,
    D_MH,
    D_RS,
    X_VALS,
    RHO_VALS,
    THETA_VALS,
    Q_POLAR_LPF,
    Q_CART_LPF,
    OMEGA_SF,
    NU_SF,  # extras
)
from src.utils.test_utils import FP_TEST_DATA_SWITCHNET, FP_TEST_DATA_PARSED

from src.data.data_io import (
    load_dir,
    load_hdf5_to_dict,
    save_dict_to_hdf5,
)

from src.utils.logging_utils import parse_val
from src.models.MFISNet_Fused import load_MFISNet_Fused_from_state_dict

tiny_debugging_dataset = "test/assets/2024-05-28_tiny_debugging_dataset"
scratch_dir = os.path.join("test", "assets", "scratch")

### Helper functions ###
# helper functions to set up the test environment
# DummyArgs to call the main function


def select_random_dir(*args, **kwargs):
    """Acts the same as select_random_file but just"""
    tmp_dir = select_random_file(*args, **kwargs)
    os.mkdir(tmp_dir)
    return tmp_dir


def select_random_file(
    base_dir: str,
    file_str_template: str,
    rng: np.random._generator.Generator = None,
) -> str:
    """Helper function to choose an unused file path
    Note: expects file_str_template to be something like "result_{rnd_code}.txt"
    with the argument name rnd_hex
    """
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    rng = np.random.default_rng() if rng is None else rng
    # print(f"Received template: {file_str_template} looking for placeholder rnd_code")
    # print(f"Should get something like {file_str_template.format(rnd_code='<demo>')}")
    tmp_fp = None
    while tmp_fp is None:
        rnd_hex = "".join([hex(a)[2:] for a in rng.integers(0, 16, size=10)])
        cand_tmp_fp = os.path.join(base_dir, file_str_template.format(rnd_code=rnd_hex))
        if not os.path.exists(cand_tmp_fp):
            tmp_fp = cand_tmp_fp
    # print(f"Temporary file selected: {tmp_fp}")
    return tmp_fp


class DummyArgs:
    def __init__(
        self, dataset_dir, wavenumber_str, model_weights_dir, results_fp, **kwargs
    ):
        """Initialize the dummy argument class for MFISNet-Fused
        Emulates the usual interface with the argparse's args object
        """
        self.data_dir_base = dataset_dir
        self.data_input_nus = wavenumber_str
        self.eval_on_test_set = False

        self.model_weights_dir = model_weights_dir
        self.train_results_fp = results_fp

        for attr_key, attr_val in kwargs.items():
            setattr(self, attr_key, attr_val)


### Write tests that check the loading is occurring as expected ###


class Test_train_mfisnet:
    def test_load_multifreq_dataset(self) -> None:
        """Test the loading function for the multi-frequency dataset"""
        global tiny_debugging_dataset
        truncate_num = 9

        wavenumbers = [4, 16]
        freq_dir_list = [
            os.path.join(tiny_debugging_dataset, f"train_measurements_nu_{nu}")
            for nu in wavenumbers
        ]
        train_dd = load_multifreq_dataset(
            freq_dir_list,
            truncate_num=truncate_num,
            noise_to_sig_ratio=0,
            nan_mode="skip",
        )

        def kv_shrinker(key, val):
            if isinstance(val, np.ndarray):
                if val.size > 1:
                    return f"{key}<shape>", val.shape
                else:
                    return key, val.item()
            elif hasattr(val, "__len__") and len(val) > 1:
                return f"{key}<len>", len(val)
            else:
                return key, val

        train_dd_short = dict(kv_shrinker(k, v) for (k, v) in train_dd.items())
        print(f"train_dd (shortened): {train_dd_short}")

        train_nu4_dd = load_dir(
            freq_dir_list[0], freq_dir_list[0], truncate_num=truncate_num
        )
        train_nu16_dd = load_dir(
            freq_dir_list[1], freq_dir_list[1], truncate_num=truncate_num
        )

        freq_dep_keys = {Q_POLAR_LPF, Q_CART_LPF, D_RS, D_MH, NU_SF, OMEGA_SF}
        for nui, train_nu_dd in enumerate([train_nu4_dd, train_nu16_dd]):
            for key in train_nu_dd.keys():
                ref_nu_val = train_nu_dd[key]
                lmfd_val = train_dd[key]  # load-multi-freq-dataset value
                if key not in freq_dep_keys:
                    if isinstance(ref_nu_val, np.ndarray) or isinstance(
                        ref_nu_val, list
                    ):
                        assert not np.any(np.isnan(lmfd_val))
                        assert np.allclose(ref_nu_val, lmfd_val)
                    else:
                        assert ref_nu_val == lmfd_val
                elif key in {NU_SF, OMEGA_SF}:
                    assert np.allclose(ref_nu_val.item(), lmfd_val[nui])
                else:
                    assert np.allclose(ref_nu_val, lmfd_val[:, nui, ...])
                # print("Good!")

    # Takes ~10 seconds for training (on 4 cores, no cuda)
    # Call pytest with --runslow to run this test
    @pytest.mark.slow
    def test_train_mfisnet_fused_basic(self) -> None:
        """Call the main training script on the tiny debugging dataset"""
        global scratch_dir, tiny_debugging_dataset
        if not os.path.exists(scratch_dir):
            os.mkdir(scratch_dir)

        # In case we need to report the seed I guess...
        rng_base = np.random.default_rng()
        rng_dirs_seed = rng_base.integers(1 << 16, size=1)
        model_seed = rng_base.integers(1 << 16, size=1)
        rng_dirs = np.random.default_rng(rng_dirs_seed)
        # model_seed = 17329 # this gives trouble if we only allow 10 epochs (for val err)
        # model_seed = 50895 # try this
        print(f"Using model seed {model_seed}")

        try:
            # Set up the directories
            base_dir = scratch_dir
            dataset_dir = tiny_debugging_dataset
            model_weights_dir = select_random_dir(
                base_dir, "model_weights_{rnd_code}", rng_dirs
            )
            results_fp = select_random_file(
                base_dir, "results_fp_{rnd_code}.txt", rng_dirs
            )
            wavenumber_str = ["4", "16"]  # two-frequency
            N_freqs = len(wavenumber_str)
            # Set up the arguments
            args = DummyArgs(
                dataset_dir,
                wavenumber_str,
                model_weights_dir,
                results_fp,
                # more results as keyword-arguments
                truncate_num=None,
                truncate_num_val=None,
                seed=model_seed,
                n_cnn_1d=3,
                n_cnn_2d=3,
                n_cnn_channels_1d=24,
                n_cnn_channels_2d=24,
                kernel_size_1d=60,
                kernel_size_2d=7,
                merge_middle_freq_channels="true",
                polar_padding="true",
                batch_size=16,
                n_epochs=10,
                lr_init=1e-3,
                weight_decay=1e-3,
                eta_min=1e-3,
                n_epochs_per_log=5,
                debug=True,
                forward_model_adjustment=1.0,
                noise_to_signal_ratio=0,
                # Actually can ignore these because the testing device
                # is not guaranteed to have access to our wandb account
                wandb_project="2024-06-04_unit_test_mfisnet_fused",
                wandb_entity="recursive-linearization",
                # wandb_mode="online",
                wandb_mode="disabled",
                big_init=True,
            )
            arg_dict = {
                field: getattr(args, field)
                for field in args.__dir__()
                if field[0] != "_"
            }
            print(f"Arguments: {arg_dict}")
            print(f"Base directory ({base_dir}) contents: {os.listdir(base_dir)}")

            # Run the main code
            start_time = time.perf_counter()
            model_orig = train_mfisnet_fused_main(args, return_model=True)
            train_time = time.perf_counter() - start_time
            print(f"Training time: {train_time:2f}s")
            print(f"Model: {model_orig}")
            print(
                f"Model parameters... {[list(p.size()) for p in model_orig.parameters()]}"
            )

            # Read in the results
            with open(results_fp, "r") as file:
                file_contents = [line.strip().split("\t") for line in file]
            header = file_contents[0]
            contents = file_contents[1:]
            file_dict = {
                field: [parse_val(line[i]) for line in contents]
                for (i, field) in enumerate(header)
            }
            print(f"Header: {header}")
            print(f"Epochs: {file_dict['epoch']}")
            print(f"Rel l2 (val):   {file_dict['eval_final_rel_l2']}")
            print(f"Rel l2 (train): {file_dict['train_rel_l2']}")

            print(
                f"Weights dir {model_weights_dir} contents: {os.listdir(model_weights_dir)}"
            )
            last_epoch = file_dict["epoch"][-1]
            model_fp = os.path.join(model_weights_dir, f"epoch_{last_epoch}.pickle")
            print(f"Loading the model weights from {model_fp}...")

            # Load the model and compare against the original model
            device = device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model_orig.to(device)  # ensure this is also on the same device
            model_state_dict = torch.load(model_fp, map_location=device)
            model_loaded = load_MFISNet_Fused_from_state_dict(
                model_state_dict, N_freqs, polar_padding=True
            )
            model_loaded = model_loaded.to(device)
            model_loaded.eval()

            # Now for the comparisons...
            # print(f"Model parameters... {list(model_orig.parameters())}")
            # Check the conv1d layers first
            for layer_orig, layer_loaded in zip(
                model_orig.conv_1d_layers, model_loaded.conv_1d_layers
            ):
                assert torch.allclose(layer_orig, layer_loaded)
            # Then the conv2d layers
            for layer_orig, layer_loaded in zip(
                model_orig.conv_2d_layers, model_loaded.conv_2d_layers
            ):
                assert torch.allclose(layer_orig.weight, layer_loaded.weight)
                assert torch.allclose(layer_orig.bias, layer_loaded.bias)

            print(f"Loaded model parameters match!")

        finally:
            # Cleanup at the very end...
            if os.path.exists(model_weights_dir):
                shutil.rmtree(model_weights_dir)
            if os.path.exists(results_fp):
                os.remove(results_fp)


if __name__ == "__main__":
    pytest.main()
