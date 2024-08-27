import pytest
import numpy as np
import os
import torch
import pandas as pd

from train_MFISNet_Refinement import (
    main as train_mfisnet_refinement_main,
    load_data,
    setup_args,
)

from src.models.MFISNet_Refinement import load_MFISNet_Refinement_from_state_dict

from src.utils.test_utils import (
    FRMT_TEST_MEAS,
    FRMT_TRAIN_MEAS,
    FRMT_VAL_MEAS,
    DIR_TEST_SCOBJ,
    DIR_TRAIN_SCOBJ,
    DIR_VAL_SCOBJ,
    WAVENUMBERS_PRESENT,
    check_arrays_close,
    EXPECTED_N_M,
    EXPECTED_N_H,
    EXPECTED_N_THETA,
    EXPECTED_N_RHO,
    fixture_tmp_dir,
)


class Test_load_data:
    def test_0(self) -> None:
        """
        Tests that the function returns without errors, and that the
        arrays saved in the MultiFreqResidData object have the expected
        shape. Also checks that loading the data a second time gives the
        same arrays. This guards against adding measurement noise when we
        don't want it.
        """
        n_samps = 4
        n_freqs = len(WAVENUMBERS_PRESENT)

        # Load the data
        data, _ = load_data(
            meas_data_dir_frmt=FRMT_TRAIN_MEAS,
            scobj_data_dir=DIR_TRAIN_SCOBJ,
            wavenumbers=WAVENUMBERS_PRESENT,
            truncate_num=n_samps,
        )

        # Check that the data has the expected shape

        assert data.d_mh.shape == (n_samps, n_freqs, EXPECTED_N_M, EXPECTED_N_H, 2)
        assert data.q_polar_lpf.shape == (
            n_samps,
            n_freqs,
            EXPECTED_N_THETA,
            EXPECTED_N_RHO,
        )

        # Load the data a second time
        data_2, _ = load_data(
            meas_data_dir_frmt=FRMT_TRAIN_MEAS,
            scobj_data_dir=DIR_TRAIN_SCOBJ,
            wavenumbers=WAVENUMBERS_PRESENT,
            truncate_num=n_samps,
        )
        # Check that the data arrays are the same
        check_arrays_close(data.d_mh.numpy(), data_2.d_mh.numpy())
        check_arrays_close(data.q_polar_lpf.numpy(), data_2.q_polar_lpf.numpy())

    def test_1(self) -> None:
        """Adds noise to the data, makes sure that the correct parts of the data are different."""

        data, _ = load_data(
            meas_data_dir_frmt=FRMT_TRAIN_MEAS,
            scobj_data_dir=DIR_TRAIN_SCOBJ,
            wavenumbers=WAVENUMBERS_PRESENT,
            truncate_num=4,
            noise_to_sig_ratio=0.1,
        )

        data_2, _ = load_data(
            meas_data_dir_frmt=FRMT_TRAIN_MEAS,
            scobj_data_dir=DIR_TRAIN_SCOBJ,
            wavenumbers=WAVENUMBERS_PRESENT,
            truncate_num=4,
            noise_to_sig_ratio=0.1,
        )

        data_no_noise, _ = load_data(
            meas_data_dir_frmt=FRMT_TRAIN_MEAS,
            scobj_data_dir=DIR_TRAIN_SCOBJ,
            wavenumbers=WAVENUMBERS_PRESENT,
            truncate_num=4,
        )

        # Check that the data arrays are different
        assert not np.allclose(data.d_mh.numpy(), data_2.d_mh.numpy())
        assert not np.allclose(data.d_mh.numpy(), data_no_noise.d_mh.numpy())

        # Check that the q arrays are the same.
        assert np.allclose(data.q_polar_lpf.numpy(), data_2.q_polar_lpf.numpy())

class Test_train_mfisnet_refinement_main:
    # Takes ~10 seconds for training (on 4 cores, no cuda)
    # Call pytest with --runslow to run this test
    @pytest.mark.slow
    def test_0(self, fixture_tmp_dir) -> None:
        """
        Tests that the main function runs without errors, and that the
        output files are created.
        """

        results_fp = os.path.join(fixture_tmp_dir, "results.txt")
        models_dir = os.path.join(fixture_tmp_dir, "models")
        output_train_dir = os.path.join(fixture_tmp_dir, "preds_train")
        output_val_dir = os.path.join(fixture_tmp_dir, "preds_val")

        n_epochs_pretrain = 1
        n_epochs_finetune = 2

        cmd_line_args = f"""
-meas_dir_train_frmt {FRMT_TRAIN_MEAS} \
-scobj_dir_train {DIR_TRAIN_SCOBJ} \
-meas_dir_val_frmt {FRMT_VAL_MEAS} \
-scobj_dir_val {DIR_VAL_SCOBJ} \
-output_dir_train {output_train_dir} \
-output_dir_val {output_val_dir} \
-results_fp {results_fp} \
-model_weights_dir {models_dir} \
-wavenumbers {' '.join(map(str, WAVENUMBERS_PRESENT))} \
-truncate_num 3 \
-truncate_num_val 3 \
-seed 2002 \
-batch_size 2 \
-n_epochs_pretrain {n_epochs_pretrain} \
-n_epochs_finetune {n_epochs_finetune} \
-lr_init 1e-3 \
-n_epochs_per_log 1 \
-dont_use_wandb
"""
        args = setup_args(cmd_line_args)
        train_mfisnet_refinement_main(args)

        expected_n_epochs = (
            len(WAVENUMBERS_PRESENT) * n_epochs_pretrain + n_epochs_finetune
        )

        # Check for the presence of the results file by loading it with pandas
        results = pd.read_table(results_fp)

        # Check for the presence of the model weights
        weights_0_fp = os.path.join(models_dir, "epoch_0.pickle")
        assert os.path.isfile(weights_0_fp), os.listdir(models_dir)

        assert len(os.listdir(models_dir)) == expected_n_epochs

        # Makes sure a model can be loaded from the weights
        state_dict = torch.load(weights_0_fp)
        model = load_MFISNet_Refinement_from_state_dict(state_dict)

        # Check for the presence of the output files
        assert os.path.isfile(
            os.path.join(output_train_dir, "scattering_objs_0.h5")
        ), os.listdir(output_train_dir)
        assert os.path.isfile(
            os.path.join(output_val_dir, "scattering_objs_0.h5")
        ), os.listdir(output_val_dir)


if __name__ == "__main__":
    pytest.main()
