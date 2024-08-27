# Load and evaluate the MFISNet_Fused on the test set

import logging
import argparse
import yaml
import os

import numpy as np
import torch

from src.data.data_io import (
    save_dict_to_hdf5,
)

# from src.FYNet.multi_freq_FYNet import FYNetMultiFreq
from src.models.MFISNet_Fused import MFISNet_Fused, load_MFISNet_Fused_from_state_dict
from src.data.data_transformations import (
    prep_conv_interp_2d,
    prep_polar_padder,
    polar_pad_and_apply,
)
from src.data.data_io import load_hdf5_to_dict, _get_number_from_filename
from src.data.data_naming_constants import KEYS_FOR_EXPERIMENT_INFO_OUT, Q_CART
from src.training_utils.make_predictions import make_preds_on_dataset
from src.training_utils.loss_functions import (
    psnr,
    relative_l2_error,
    _mse_along_batch,
)
from train_MFISNet_Fused import load_multifreq_dataset, setup_single_dataset


from src.utils.logging_utils import FMT, TIMEFMT, find_best_epoch

SCOBJ_DIR_TEST = "test_scattering_objs"


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir_base",
        type=str,
        help="Indicate the directory containing all the measurement folders"
        " corresponding to the relevant frequencies and data subsets",
    )
    parser.add_argument("--data_input_nus", type=str, nargs="+")
    parser.add_argument("--eval_on_test_set", default=True, action="store_true")
    parser.add_argument(
        "--no_eval_on_test_set", action="store_false", dest="eval_on_test_set"
    )

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
    # parser.add_argument(
    #     "--hyperparam_summary_fp",
    #     type=str,
    #     help="Point to the hyperparameter search summary file (yaml format)",
    # )
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

    parser.add_argument("--seed", default=None, type=int)  # seed bc we're using noise
    parser.add_argument(
        "--noise_to_signal_ratio", default=None, type=float
    )  # test with noise

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="For evaluation this just needs to be small enough to fit into GPU memory",
    )
    # parser.add_argument("--n_epochs_per_log", type=int, default=5)
    parser.add_argument("--debug", default=False, action="store_true")
    a = parser.parse_args()
    return a


class LinearData(torch.utils.data.Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.X = X
        self.y = y
        logging.info(
            "Initialized a LinearData instance with X shape: %s and y shape: %s",
            self.X.shape,
            self.y.shape,
        )
        self.n_samples = X.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main(args: argparse.Namespace) -> None:
    """
    1. Load data
    2. Do necessary transformations
    3. Set up NN with hyperparameters
    4. Run and evaluate the NN on the test set
    """
    if not os.path.isdir(args.test_output_predictions_dir):
        os.mkdir(args.test_output_predictions_dir)
    # Set seeds for reproducible noise
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Find the best epoch
    best_epoch_dd = find_best_epoch(
        args.training_results_fp, args.training_results_key, selection_mode="min"
    )
    best_epoch = best_epoch_dd["epoch"]
    hps_polar_padding = best_epoch_dd["polar_padding"]
    logging.info(f"Best epoch: {best_epoch}")

    model_fp = os.path.join(args.model_dir, args.model_fp_format.format(best_epoch))

    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Evaluating on device: %s", device)

    #########################################################
    # Load data
    data_dir_base = args.data_dir_base
    str_nu_list = (
        args.data_input_nus
    )  # nu in string form (to preserve decimals properly)
    nu_list = [float(str_nu) for str_nu in str_nu_list]
    N_freqs = len(nu_list)
    logging.info(f"data_dir_base: {data_dir_base}")
    logging.info(f"nu values received: {str_nu_list}")

    # args.eval_on_test_set = False

    if args.eval_on_test_set:
        eval_set_name = "test"
    else:
        eval_set_name = "val"
    eval_files = [
        os.path.join(data_dir_base, f"{eval_set_name}_measurements_nu_{nu}")
        for nu in str_nu_list
    ]

    logging.info(f"Attempting to load the {eval_set_name} set: {eval_files}")

    ### Load Evaluation dataset to a dictionary and local variables ###
    logging.info(f"Loading evaluation dataset")
    eval_dd = load_multifreq_dataset(
        eval_files,
        # key_replacement=key_replacement,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_mh",
        nan_mode="keep",
        load_cart=True,
    )
    eval_q_polar = eval_dd["q_polar"]
    eval_q_cart = eval_dd["q_cart"]
    eval_d_mh = eval_dd["d_mh"]

    rho_vals = eval_dd["rho_vals"]
    theta_vals = eval_dd["theta_vals"]
    h_vals = eval_dd["h_vals"]
    omega_vals = eval_dd["omega_sf"]
    x_vals = (
        eval_dd["x_vals"]
        if "x_vals" in eval_dd.keys()
        else np.linspace(-0.5, 0.5, eval_q_cart.shape[-1])
    )  # default value..
    N_x = x_vals.shape[0]
    N_rho = rho_vals.shape[0]
    N_h = h_vals.shape[0]
    N_theta = theta_vals.shape[0]
    N_eval = eval_q_polar.shape[0]

    # Prepare the LinearData object
    eval_dset = setup_single_dataset(eval_dd["q_polar"], eval_dd["d_mh"])

    # Prepare the DataLoader
    eval_dloader = torch.utils.data.DataLoader(
        eval_dset, batch_size=args.batch_size, shuffle=False
    )

    ##### Load model from disk #####
    model_state_dict = torch.load(model_fp, map_location=device)
    model = load_MFISNet_Fused_from_state_dict(
        model_state_dict, N_freqs, polar_padding=hps_polar_padding
    )
    model = model.to(device)
    model.eval()

    logging.info(f"Loaded model: {model}")

    #########################################################
    # Make the predictions and save them to disk.

    experiment_info = {}
    for key, value in eval_dd.items():
        if key in KEYS_FOR_EXPERIMENT_INFO_OUT:
            experiment_info[key] = value

    make_preds_on_dataset(
        model=model,
        dloader=eval_dloader,
        experiment_info=experiment_info,
        output_dir=args.test_output_predictions_dir,
        device=device,
        shard_size=500,
    )

    #########################################################
    # Load the predictions to evaluate
    scobj_dir_test = os.path.join(args.data_dir_base, SCOBJ_DIR_TEST)

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

    # logging.info(f"Main loop successful")
    logging.info(f"~~~Summary~~~")
    logging.info(f"MSE error: {cart_mse_mean:.3e}±{cart_mse_std:.3e}")
    logging.info(f"Rel l2 error: {cart_rel_l2_mean:.5f}±{cart_rel_l2_std:.5f}")
    logging.info(f"PSNR: {cart_psnr_mean:.5f}±{cart_psnr_std:.5f}")

    common_settings_dict = {
        # Grid info
        "N_rho": N_rho,
        "N_theta": N_theta,
        "N_m": N_theta,
        "N_h": N_h,
        "N_x": N_x,
        "N_freqs": N_freqs,
        # Hyperparam info
        "N_cnn_1d": model.N_cnn_1d,
        "N_cnn_2d": model.N_cnn_2d,
        "N_channels_cnn_1d": model.c_1d,
        "N_channels_cnn_2d": model.c_2d,
        "kernel_size_1d": model.w_1d,
        "kernel_size_2d": model.w_2d,
    }
    summary_errors_dict = {
        "cart_mse_mean": cart_mse_mean,
        "cart_mse_std": cart_mse_std,
        "cart_rel_l2_mean": cart_rel_l2_mean,
        "cart_rel_l2_std": cart_rel_l2_std,
        "cart_psnr_mean": cart_psnr_mean,
        "cart_psnr_std": cart_psnr_std,
    }
    # common_settings_dict = {key: val for (key,val) in common_settings_dict.items()}
    summary_errors_dict = {
        key: val.item() for (key, val) in summary_errors_dict.items()
    }

    summary_dict = {
        # Summary values
        **summary_errors_dict,
        # Metadata
        "model_file_name": model_fp,
        "predictions_fp": args.test_output_predictions_dir,
        **common_settings_dict,
    }

    # Save to disk
    with open(args.test_output_summary_fp, "w") as sfile:
        yaml.dump(summary_dict, sfile, default_flow_style=False)
    logging.info(f"Saved summary file to {args.test_output_summary_fp}")
    logging.info(f"Saved predictions to {args.test_output_predictions_dir}")
    logging.info(f"Finished!")


if __name__ == "__main__":
    a = setup_args()

    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    if a.debug:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)

    logging.info(f"Received the following arguments: {a}")
    main(a)
