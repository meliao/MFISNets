# Load and evaluate the MFISNet-Refinement on the test set

import logging
from typing import Tuple, Dict, List, Callable
import argparse
import yaml
import os
import sys

import numpy as np
import torch

from src.models.MFISNet_Refinement import (
    MFISNet_Refinement,
    load_MFISNet_Refinement_from_state_dict,
)
from src.data.data_io import load_hdf5_to_dict, _get_number_from_filename

from src.training_utils.make_predictions import make_preds_on_dataset
from src.training_utils.loss_functions import (
    psnr,
    relative_l2_error,
    _mse_along_batch,
)
from src.utils.logging_utils import FMT, TIMEFMT, find_best_epoch
from src.data.data_naming_constants import (
    X_VALS,
    Q_CART,
)
from train_MFISNet_Refinement import load_data

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir_base",
        type=str,
        help="Indicate the directory containing all the measurement folders"
        " corresponding to the relevant frequencies and data subsets",
    )
    parser.add_argument("--wavenumbers", type=str, nargs="+")
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Point to the directory containing the desired model parameters",
    )
    parser.add_argument(
        "--training_results_fp",
        type=str,
        help="tab-separated file containing training results.",
    )
    parser.add_argument(
        "--training_results_key",
        type=str,
        help="key used to select the epoch with the minimal validation loss.",
    )
    parser.add_argument("--model_fp_format", type=str, default="epoch_{}.pickle")

    parser.add_argument(
        "--hyperparam_summary_fp",
        type=str,
        help="Point to the hyperparameter search summary file (yaml format)",
    )
    parser.add_argument(
        "--test_output_summary_fp",
        type=str,
        help="Point to the desired output summary file",
    )
    parser.add_argument(
        "--test_output_predictions_dir",
        type=str,
        help="Point to the desired output predictions file",
    )
    parser.add_argument("--noise_to_signal_ratio", default=None, type=float)
    parser.add_argument("--type_C_model", default=False, action="store_true")

    parser.add_argument("--debug", default=False, action="store_true")
    a = parser.parse_args()
    return a


SCOBJ_DIR_TEST = "test_scattering_objs"
MEAS_DIR_FRMT_TEST = "test_measurements_nu_{}"


def main(args: argparse.Namespace) -> None:
    """
    1. Load data
    2. Do necessary transformations
    3. Set up NN with hyperparameters
    4. Run and evaluate the NN on the test set
    """
    if not os.path.isdir(args.test_output_predictions_dir):
        os.mkdir(args.test_output_predictions_dir)

    if args.hyperparam_summary_fp is not None:
        # Load hyperparameter summary:
        with open(args.hyperparam_summary_fp, "r") as hsf:
            hyperparam_sd = yaml.load(hsf, Loader=yaml.Loader)
        # hyperparam_sd = yaml.load(args.hyperparam_summary_fp, Loader=yaml.Loader)
        logging.debug(f"hp_sd: {hyperparam_sd.keys()}")
        hps_log_info = hyperparam_sd["log_info"]
        logging.debug(f"hps log info: {hps_log_info}", flush=True)
        batch_size = hps_log_info["batch_size"]
    else:
        batch_size = 32

    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Evaluating on device: %s", device)

    N_freqs = len(args.wavenumbers)
    logging.info(f"data_dir_base: {args.data_dir_base}")
    logging.info("nu values received: %s", args.wavenumbers)

    #########################################################
    # Load the dataset
    meas_dir_test_frmt = os.path.join(args.data_dir_base, MEAS_DIR_FRMT_TEST)
    scobj_dir_test = os.path.join(args.data_dir_base, SCOBJ_DIR_TEST)
    test_dset, test_metadata_dd = load_data(
        meas_dir_test_frmt,
        scobj_dir_test,
        args.wavenumbers,
        None,
        args.noise_to_signal_ratio,
    )
    test_dset.output_nan_samples_bool = True

    test_dloader = torch.utils.data.DataLoader(
        test_dset, batch_size=batch_size, shuffle=False
    )

    #########################################################
    # Load the model

    best_epoch_dd = find_best_epoch(
        args.training_results_fp, args.training_results_key, selection_mode="min"
    )
    epoch_num = best_epoch_dd["epoch"]
    model_fp = os.path.join(args.model_dir, args.model_fp_format.format(epoch_num))

    model_state_dict = torch.load(model_fp, map_location=device)
    model = load_MFISNet_Refinement_from_state_dict(model_state_dict)
    model = model.to(device)
    model.freq_pred_idx = N_freqs - 1  # Set to the highest frequency
    model.eval()

    logging.info(f"Loaded model: {model}")

    #########################################################
    # Make the predictions and save them to the disk

    logging.info("Starting to make predictions on the test set")
    logging.info("Saving predictions to %s", args.test_output_predictions_dir)

    make_preds_on_dataset(
        model=model,
        dloader=test_dloader,
        experiment_info=test_metadata_dd,
        output_dir=args.test_output_predictions_dir,
        device=device,
        shard_size=500,
    )

    #########################################################
    # Load the predictions to evaluate

    preds_file_lst = os.listdir(args.test_output_predictions_dir)
    preds_file_lst = sorted(preds_file_lst, key=_get_number_from_filename)

    test_preds_lst = [
        load_hdf5_to_dict(os.path.join(args.test_output_predictions_dir, x))
        for x in preds_file_lst
    ]
    test_preds_arr = np.concatenate([x[Q_CART] for x in test_preds_lst])

    logging.info("Loaded test predictions with shape %s", test_preds_arr.shape)

    test_targets_files = os.listdir(scobj_dir_test)
    test_targets_files = sorted(test_targets_files, key=_get_number_from_filename)
    test_targets_lst = [os.path.join(scobj_dir_test, x) for x in test_targets_files]
    test_targets_arr = np.concatenate(
        [load_hdf5_to_dict(x)[Q_CART] for x in test_targets_lst]
    )
    logging.info("Loaded test targets with shape %s", test_targets_arr.shape)
    n_samples = test_targets_arr.shape[0]

    #########################################################
    # Evaluate the predictions
    test_preds_arr = torch.from_numpy(test_preds_arr)
    test_targets_arr = torch.from_numpy(test_targets_arr)

    #########################################################
    # Remove nans if necessary
    is_nan_preds = torch.isnan(test_preds_arr[:, 0, 0])
    is_not_nan = torch.logical_not(is_nan_preds)
    logging.info(
        "Removing %i samples that have nans", n_samples - torch.sum(is_not_nan)
    )

    test_preds_arr = test_preds_arr[is_not_nan]
    test_targets_arr = test_targets_arr[is_not_nan]

    rel_l2_errors = relative_l2_error(
        preds=test_preds_arr,
        targets=test_targets_arr,
    ).numpy()

    mse_errors = _mse_along_batch(
        preds=test_preds_arr, targets=test_targets_arr
    ).numpy()
    psnrs = psnr(preds=test_preds_arr, targets=test_targets_arr).numpy()
    cart_rel_l2_mean = np.mean(rel_l2_errors)
    cart_rel_l2_std = np.std(rel_l2_errors)
    cart_mse_mean = np.mean(mse_errors)
    cart_mse_std = np.std(mse_errors)
    cart_psnr_mean = np.mean(psnrs)
    cart_psnr_std = np.std(psnrs)

    logging.info(f"~~~Summary~~~")
    logging.info(f"MSE error: {cart_mse_mean:.3e}±{cart_mse_std:.3e}")
    logging.info(f"Rel l2 error: {cart_rel_l2_mean:.5f}±{cart_rel_l2_std:.5f}")
    logging.info(f"PSNR: {cart_psnr_mean:.5f}±{cart_psnr_std:.5f}")

    summary_errors_dict = {
        "cart_mse_mean": cart_mse_mean,
        "cart_mse_std": cart_mse_std,
        "cart_rel_l2_mean": cart_rel_l2_mean,
        "cart_rel_l2_std": cart_rel_l2_std,
        "cart_psnr_mean": cart_psnr_mean,
        "cart_psnr_std": cart_psnr_std,
    }

    summary_errors_dict = {k: v.item() for k, v in summary_errors_dict.items()}

    summary_dict = {
        # Summary values
        **summary_errors_dict,
        # Metadata
        "wavenumbers": args.wavenumbers,
        "data_dir_base": args.data_dir_base,
        "model_file_name": model_fp,
        "hyperparam_summary_file_name": args.hyperparam_summary_fp,
    }

    # Save to disk
    with open(args.test_output_summary_fp, "w") as sfile:
        yaml.dump(summary_dict, sfile, default_flow_style=False)
    logging.info(f"Saved summary file to {args.test_output_summary_fp}")
    logging.info(f"Finished!")


if __name__ == "__main__":
    a = setup_args()

    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    if a.debug:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)

    main(a)
