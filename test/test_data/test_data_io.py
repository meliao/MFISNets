import pytest
import numpy as np
import os
from src.data.data_io import (
    # load_SwitchNet_file_type,
    # concat_dir_of_parsed_files,
    # load_parsed_file_type,
    load_dir,
    load_hdf5_to_dict,
    save_dict_to_hdf5,
    save_field_in_hdf5,
    load_field_in_hdf5,
    update_field_in_hdf5
)
import numpy as np
from src.data.data_naming_constants import (
    Q_CART, Q_POLAR, D_MH, D_RS, X_VALS, RHO_VALS, THETA_VALS
)
from src.utils.test_utils import FP_TEST_DATA_SWITCHNET, FP_TEST_DATA_PARSED

# Outdated...
EXPECTED_SWITCHNET_KEYS = ["Input", "Output", "x_vals", "theta_vals", "omega_vals"]
EXPECTED_PARSED_FILE_KEYS = [
    "Input",
    "Output",
    "Rho_vals",
    "Theta_vals",
    "H_vals",
    "Omega_vals",
]

# Test on a tiny simplified version of the true dataset structure
tiny_debugging_dataset = "test/assets/2024-05-28_tiny_debugging_dataset"
scratch_dir = os.path.join("test", "assets", "scratch")

class Test_tiny_debugging_dataset:
    def test_load_dir(self: None) -> None:
        """try to load the tiny debugging dataset for train_measurements_nu_4 with load_dir"""
        train_scobj_dir = os.path.join(tiny_debugging_dataset, "train_scattering_objs")
        train_nu_4_dir = os.path.join(tiny_debugging_dataset, "train_measurements_nu_4")
        train_nu_4_dd = load_dir(train_nu_4_dir, train_scobj_dir, truncate_num=7, load_cart_and_rs=True)

        print(f"Received keys: {train_nu_4_dd.keys()}")
        print(f"Received shapes: q_cart {train_nu_4_dd[Q_CART].shape} q_polar {train_nu_4_dd[Q_POLAR].shape}")
        print(f"Received shapes: d_rs {train_nu_4_dd[D_RS].shape} d_mh {train_nu_4_dd[D_MH].shape}")
        print(f"Received shapes: x_vals {train_nu_4_dd[X_VALS].shape} rho_vals "
              f"{train_nu_4_dd[RHO_VALS].shape} theta_vals {train_nu_4_dd[THETA_VALS].shape}")

        assert train_nu_4_dd[Q_CART].shape[0] == train_nu_4_dd[Q_POLAR].shape[0]
        assert train_nu_4_dd[D_RS].shape[0] == train_nu_4_dd[D_MH].shape[0]

        # Check this matches the result when we load only one file..
        single_block_fp = os.path.join(train_nu_4_dir, "measurements_0.h5")
        train_nu_4_block1_dd = load_hdf5_to_dict(single_block_fp)

        block_len = train_nu_4_block1_dd[Q_CART].shape[0]
        assert np.allclose(train_nu_4_dd[Q_CART][:block_len], train_nu_4_block1_dd[Q_CART])
        assert np.allclose(train_nu_4_dd[D_RS][:block_len], train_nu_4_block1_dd[D_RS])


    def test_load_dir_wout_qcart(self: None) -> None:
        """try to load the tiny debugging dataset for train_measurements_nu_4 with load_dir but without loading q_cart or d_rs"""
        train_scobj_dir = os.path.join(tiny_debugging_dataset, "train_scattering_objs")
        train_nu_4_dir = os.path.join(tiny_debugging_dataset, "train_measurements_nu_4")
        train_nu_4_dd = load_dir(train_nu_4_dir, train_scobj_dir, truncate_num=8, load_cart_and_rs=False)

        print(f"Received keys: {train_nu_4_dd.keys()}")
        print(f"Received shapes: x_vals {train_nu_4_dd[X_VALS].shape} rho_vals "
              f"{train_nu_4_dd[RHO_VALS].shape} theta_vals {train_nu_4_dd[THETA_VALS].shape}")

        assert Q_CART not in train_nu_4_dd.keys()
        assert D_RS not in train_nu_4_dd.keys()

class Test_hdf5_utilities_part_1:
    def _select_random_file(self: None, rng: np.random._generator.Generator = None) -> str:
        """Helper function to choose an unused file path"""
        global scratch_dir
        if not os.path.exists(scratch_dir):
            os.mkdir(scratch_dir)
        rng = np.random.default_rng() if rng is None else rng
        tmp_fp = None
        while tmp_fp is None:
            rnd_hex = "".join([hex(a)[2:] for a in rng.integers(0, 16, size=10)])
            cand_tmp_fp = os.path.join(scratch_dir, f"random_test_file_{rnd_hex}.h5")
            if not os.path.exists(cand_tmp_fp):
                tmp_fp = cand_tmp_fp
        print(f"Temp file selected: {tmp_fp}")
        return tmp_fp

    # Test the file-wide save/load using temporary and random content
    def test_random_file(self: None) -> None:
        """Test that creates a small file with random contents
        to test the saving/loading capabilities
        """
        # Fetch a file name
        rng = np.random.default_rng()
        tmp_fp = self._select_random_file(rng)
        print(f"(test random file) Writing to tmp file: {tmp_fp}")

        try:
            # Set up the file contents
            f1_shape = rng.integers(1, 16, size=3)
            f1_value = rng.normal(size=f1_shape)
            f2_shape = rng.integers(1, 16, size=4)
            f2_value = rng.normal(size=f2_shape)
            ref_dd = {"f1": f1_value, "f2": f2_value}

            # Save file to the chosen file path
            krd_saving = {"f2": "f2-stored"} # modify the keys just to test that it works
            save_dict_to_hdf5(ref_dd, tmp_fp, krd_saving)

            # Load then verify the contents match
            krd_loading = {"f2-stored": "f2"} # modify the keys just to test that it works
            rec_dd = load_hdf5_to_dict(tmp_fp, krd_loading)
            assert "f2" in rec_dd.keys()
            assert np.allclose(rec_dd["f1"], ref_dd["f1"])
            assert np.allclose(rec_dd["f2"], ref_dd["f2"])
        # except Exception as e:
        #     print(f"test_field_slices encountered error: {e}")
        #     raise e
        finally:
            # Clean up
            if os.path.exists(tmp_fp):
                os.remove(tmp_fp)

    def test_field_slices(self: None) -> None:
        """Test the field-only modifications"""
        # Fetch a file name
        rng = np.random.default_rng()
        tmp_fp = self._select_random_file(rng)
        print(f"(test field slices) Writing to tmp file: {tmp_fp}")
        try:
            # Set up the file contents
            update_len = 5
            f1_shape = [10, *rng.integers(1, 16, size=2)]
            f1_value = rng.normal(size=f1_shape)
            f2_shape = rng.integers(1, 16, size=4)
            f2_value = rng.normal(size=f2_shape)
            f1_perturbation = rng.normal(size=f1_shape)[:update_len]
            f1_value_new = np.copy(f1_value)
            f1_value_new[:update_len] += f1_perturbation

            ref1_dd = {"f1": f1_value, "f2": f2_value}
            ref2_dd = {"f1": f1_value_new, "f2": f2_value}

            # Start by only saving one field
            save_dict_to_hdf5({"f2": f2_value}, tmp_fp)
            save_field_in_hdf5("f1", f1_value, tmp_fp, overwrite=True)

            # Load just the f1 entry:
            rec_f1_orig = load_field_in_hdf5("f1", tmp_fp)
            assert np.allclose(rec_f1_orig, f1_value)
            rec_f1_update = rec_f1_orig[:update_len] + f1_perturbation
            update_field_in_hdf5("f1", rec_f1_update, tmp_fp, idx_slice=slice(0,update_len))

            # Verify everything looks right..
            rec_new_dd = load_hdf5_to_dict(tmp_fp)
            assert np.allclose(rec_new_dd["f1"], ref2_dd["f1"])
            assert np.allclose(rec_new_dd["f2"], ref2_dd["f2"])
        # except Exception as e:
        #     print(f"test_field_slices encountered error: {e}")
        #     raise e
        finally:
            # Clean up
            if os.path.exists(tmp_fp):
                os.remove(tmp_fp)

if __name__ == "__main__":
    pytest.main()
