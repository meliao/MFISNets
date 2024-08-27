"""
This script does the following:
1. Loads data from specified directories
2. Sets up an MFISNet-Parallel model (with or without a skip connection)
3. Trains the model
4. Runs inference on all of the train / val samples.
5. Transforms the intermediate outputs to Cart coordinates.
6. Writes the scattering objects to disk
"""

import logging
from typing import Tuple, Dict, List, Callable
import argparse
from timeit import default_timer
import os
import shlex
import sys
import numpy as np
import h5py
import torch
import pandas as pd
import wandb

from src.data.data_naming_constants import (
    X_VALS,
    RHO_VALS,
    THETA_VALS,
    Q_POLAR,
    Q_POLAR_LPF,
    Q_CART,
    D_MH,
    KEYS_FOR_TRAINING_METADATA,
    D_RS,
    SAMPLE_COMPLETION,
    H_VALS,
    M_VALS,
    KEYS_FOR_EXPERIMENT_INFO_OUT,
)


from src.data.data_io import load_dir
from src.models.MFISNet_Parallel import MFISNet_Parallel
from src.training_utils.train_loop import train, evaluate_losses_on_dataloader
from src.training_utils.loss_functions import (
    MSEModule,
)
from src.training_utils.make_predictions import make_preds_on_dataset
from src.utils.logging_utils import FMT, TIMEFMT, write_result_to_file, hash_dict
from src.data.add_noise import add_noise_to_d


def setup_args(argument_string: str = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-meas_dir_train_frmt")
    parser.add_argument("-scobj_dir_train")
    parser.add_argument("-meas_dir_val_frmt")
    parser.add_argument("-scobj_dir_val")
    parser.add_argument("-output_dir_train")
    parser.add_argument("-output_dir_val")
    parser.add_argument("-results_fp")
    parser.add_argument("-model_weights_dir")
    parser.add_argument("-wavenumbers", nargs="+")
    parser.add_argument("-truncate_num", type=int)
    parser.add_argument("-truncate_num_val", type=int)
    parser.add_argument("-seed", type=int, default=35675)
    parser.add_argument("-n_cnn_1d", type=int, default=3)
    parser.add_argument("-n_cnn_2d", type=int, default=3)
    parser.add_argument("-n_cnn_channels_1d", type=int, default=10)
    parser.add_argument("-n_cnn_channels_2d", type=int, default=10)
    parser.add_argument("-kernel_size_1d", type=int, default=13)
    parser.add_argument("-kernel_size_2d", type=int, default=13)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-n_epochs", type=int, default=10)
    parser.add_argument("-lr_init", type=float, default=1.0)
    parser.add_argument("-weight_decay", type=float, default=0.0)
    parser.add_argument("-eta_min", type=float, default=1e-04)
    parser.add_argument("-n_epochs_per_log", type=int, default=5)
    parser.add_argument("-debug", default=False, action="store_true")
    parser.add_argument("-wandb_project_name", type=str, help="W&B project name")
    parser.add_argument("-wandb_entity", type=str, help="The W&B entity")
    parser.add_argument("-wandb_mode", choices=["offline", "online"], default="offline")
    parser.add_argument("-dont_use_wandb", default=False, action="store_true")
    parser.add_argument("-skip_reader_setup", default=False, action="store_true")
    parser.add_argument("-dont_write_outputs", default=False, action="store_true")
    parser.add_argument("-no_filtering_bool", default=False, action="store_true")
    parser.add_argument("-noise_to_signal_ratio", default=None, type=float)

    if argument_string is None:
        # Parse the arguments from the system's argv
        args = parser.parse_args()
    else:
        # Split the argument string into a list using shlex.split()
        arg_list = shlex.split(argument_string)
        args = parser.parse_args(arg_list)
        # Adding a hash because this doesn't exist in the argument string but is
        # necessary in the main function
        id_hash = hash_dict(vars(args))
        args.hash = id_hash

    return args


class MultiFreqData(torch.utils.data.Dataset):
    def __init__(
        self,
        d_mh: torch.Tensor,
        q_polar: torch.Tensor,
        output_nan_samples_bool: bool = False,
    ) -> None:
        self.d_mh = torch.view_as_real(d_mh).to(torch.float)
        self.q_polar = q_polar.to(torch.float)
        self.output_nan_samples_bool = output_nan_samples_bool
        logging.debug("MultiFreqData: q_polar shape: %s", self.q_polar.shape)
        logging.debug("MultiFreqData: d_mh shape: %s", self.d_mh.shape)

        nan_samples = torch.any(torch.isnan(self.d_mh[:, :, 0, 0, 0]), dim=1)
        self.keep_indices = torch.argwhere(torch.logical_not(nan_samples)).flatten()
        logging.debug(
            "MultiFreqData: nan_samples has shape: %s  and sum %s",
            nan_samples.shape,
            torch.sum(nan_samples),
        )
        logging.debug(
            "MultiFreqData: self.keep_indices: shape: %s", self.keep_indices.shape
        )

        logging.info(
            "Initialized a MultiFreqData instance with d_mh shape: %s and q_polar shape: %s",
            self.d_mh.shape,
            self.q_polar.shape,
        )

        self.n_samples = self.d_mh.shape[0]

    def __len__(self):
        if self.output_nan_samples_bool:
            return self.n_samples
        else:
            return self.keep_indices.shape[0]

    def __getitem__(self, idx):
        """Want to return x with shape (n_theta, n_rho, 3).
        [:, :, 0] is the scattering object
        and [:, :, 1:3] is the real and imag parts of the
        wave field difference
        """
        if not self.output_nan_samples_bool:
            idx = self.keep_indices[idx]

        x = self.d_mh[idx]
        return x, self.q_polar[idx], self.q_polar[idx]


def load_data(
    meas_data_dir_frmt: str,
    scobj_data_dir: str,
    wavenumbers: List[str],
    truncate_num: int = None,
    noise_to_sig_ratio: float = None,
) -> MultiFreqData:
    """This function loads all of the samples indicated by the data_dir.
    It arranges the data and returns a MultiFreqData object.

    Args:
        data_dir (str): Where the data is located
        truncate_num (int, optional): _description_. Defaults to None.

    Returns:
        MultiFreqData: _description_
    """

    dd_0 = load_dir(
        meas_data_dir_frmt.format(wavenumbers[0]),
        scobj_data_dir,
        truncate_num=truncate_num,
    )

    d_mh = np.empty(
        (
            dd_0[D_MH].shape[0],
            len(wavenumbers),
            dd_0[D_MH].shape[1],
            dd_0[D_MH].shape[2],
        ),
        dtype=np.complex64,
    )
    d_mh[:, 0] = dd_0[D_MH]
    for i, w in enumerate(wavenumbers[1:], start=1):
        dd_i = load_dir(
            meas_data_dir_frmt.format(w),
            scobj_data_dir,
            truncate_num=truncate_num,
        )
        d_mh[:, i] = dd_i[D_MH]

    q_polar = torch.from_numpy(dd_0[Q_POLAR])

    if noise_to_sig_ratio is not None:
        logging.info("Adding noise with noise-to-signal ratio: %f", noise_to_sig_ratio)
        d_mh = add_noise_to_d(d_mh, noise_to_sig_ratio)

    d_mh = torch.from_numpy(d_mh)

    dset_obj = MultiFreqData(d_mh, q_polar, output_nan_samples_bool=False)
    metadata_dd = {i: dd_0[i] for i in KEYS_FOR_EXPERIMENT_INFO_OUT}
    return dset_obj, metadata_dd


def main(args: argparse.Namespace) -> None:
    """
    1. Load data
    2. Do necessary transformations
    3. Set up NN
    4. Train NN
    """
    # Make sure output directories exist
    d = [
        args.model_weights_dir,
    ]
    if args.output_dir_train is not None:
        d.append(args.output_dir_train)
    if args.output_dir_val is not None:
        d.append(args.output_dir_val)
    for i in d:
        if not os.path.isdir(i):
            os.mkdir(i)

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data from disk
    dset_train, train_metadata_dd = load_data(
        args.meas_dir_train_frmt,
        args.scobj_dir_train,
        args.wavenumbers,
        args.truncate_num,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
    )
    dset_val, val_metadata_dd = load_data(
        args.meas_dir_val_frmt,
        args.scobj_dir_val,
        args.wavenumbers,
        args.truncate_num_val,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
    )

    # Grab some metadata
    x_vals = train_metadata_dd[X_VALS]
    rho_vals = train_metadata_dd[RHO_VALS]
    theta_vals = train_metadata_dd[THETA_VALS]
    N_x = x_vals.shape[0]
    N_rho = rho_vals.shape[0]
    N_theta = theta_vals.shape[0]
    N_freqs = len(args.wavenumbers)

    n_train = len(dset_train)
    n_val = len(dset_val)

    args.n_train = n_train
    args.n_val = n_val

    logging.info("Done Data Loading. N_train=%i, N_val=%i", n_train, n_val)

    #########################################################
    # Set dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training on device: %s", device)

    train_dloader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size)
    val_dloader = torch.utils.data.DataLoader(dset_val, batch_size=args.batch_size)

    #########################################################
    # Set up NN architecture

    model = MFISNet_Parallel(
        N_h=N_rho,
        N_rho=N_rho,
        c_1d=args.n_cnn_channels_1d,
        c_2d=args.n_cnn_channels_2d,
        w_1d=args.kernel_size_1d,
        w_2d=args.kernel_size_2d,
        N_cnn_1d=args.n_cnn_1d,
        N_cnn_2d=args.n_cnn_2d,
        N_freqs=len(args.wavenumbers),
    )

    logging.info("Set up model: %s", model)

    ##########################################################################################
    # TRAIN FYNET

    loss_module_0 = MSEModule()

    loss_fn_dd = {
        "mse": loss_module_0.mse,
        "psnr": loss_module_0.psnr,
        "rel_l2": loss_module_0.relative_l2_error,
        "final_mse": loss_module_0.mse_against_final,
        "final_psnr": loss_module_0.psnr_against_final,
        "final_rel_l2": loss_module_0.relative_l2_error_against_final,
    }

    training_part = "pretrain_0"

    epoch_stagger = 0
    n_epochs_this_step = args.n_epochs

    # id_hash = hash_dict(vars(args))
    id_hash = args.hash

    def log_function(model_0, epoch_local):
        """
        NEED TO SET:
        loss_fn_dd
        epoch_stagger
        """
        with torch.no_grad():
            epoch_eff = epoch_stagger + epoch_local

            # 1. Evaluate on train set
            train_loss_dd = evaluate_losses_on_dataloader(
                model_0, train_dloader, loss_fn_dd, device
            )
            # 2. Evaluate on val set
            val_loss_dd = evaluate_losses_on_dataloader(
                model_0, val_dloader, loss_fn_dd, device
            )

            weight_norm = torch.norm(
                torch.cat([x.view(-1) for x in model_0.parameters()]), 2
            )

            # 3. Log to console and log file
            logging.info(
                "%s Epoch %i/%i. Train MSE: %f, Train Rel L2: %f, Train PSNR: %f",
                training_part,
                epoch_local,
                n_epochs_this_step,
                torch.mean(train_loss_dd["mse"]).item(),
                torch.mean(train_loss_dd["rel_l2"]).item(),
                torch.mean(train_loss_dd["psnr"]).item(),
                # train_rel_l2_aaa,
                # train_psnr_aaa,
            )
            logging.info(
                "\t Val MSE: %f, Val Rel L2: %f, Val PSNR: %f",
                torch.mean(val_loss_dd["mse"]).item(),
                torch.mean(val_loss_dd["rel_l2"]).item(),
                torch.mean(val_loss_dd["psnr"]).item(),
            )
            logging.info("\t Weight L2 norm: %f", weight_norm.item())
            train_dd = {
                # Optimization info
                "epoch": epoch_local + epoch_stagger,
                "weight_norm": weight_norm.item(),
                "training_part": training_part,
                # Experiment info
                "n_train": n_train,
                "n_val": n_val,
                "n_cnn_1d": args.n_cnn_1d,
                "n_cnn_2d": args.n_cnn_2d,
                "n_cnn_channels_1d": args.n_cnn_channels_1d,
                "n_cnn_channels_2d": args.n_cnn_channels_2d,
                "kernel_size_1d": args.kernel_size_1d,
                "kernel_size_2d": args.kernel_size_2d,
                "lr_init": args.lr_init,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "eta_min": args.eta_min,
                "n_rho_vals": N_rho,
                "n_theta_vals": N_theta,
                "hash": id_hash,
                "meas_dir_train_frmt": args.meas_dir_train_frmt,
                "scobj_dir_train": args.scobj_dir_train,
                "meas_dir_val_frmt": args.meas_dir_val_frmt,
                "scobj_dir_val": args.scobj_dir_val,
                "output_dir_train": args.output_dir_train,
                "output_dir_val": args.output_dir_val,
            }
            for k, v in train_loss_dd.items():
                train_dd["train_" + k] = torch.mean(v).item()
            for k, v in val_loss_dd.items():
                train_dd["val_" + k] = torch.mean(v).item()
            write_result_to_file(args.results_fp, **train_dd)

            # Try to log results to W&B
            if not args.dont_use_wandb:
                try:
                    wandbrun.log(train_dd)
                except ValueError:
                    logging.error("Error: wandb logging failed for %s" % wandbrun.id)

        fp_weights = os.path.join(args.model_weights_dir, f"epoch_{epoch_eff}.pickle")
        torch.save(model_0.state_dict(), fp_weights)
        model_0 = model_0.to(device)

    model = train(
        model=model,
        n_epochs=args.n_epochs,
        lr_init=args.lr_init,
        weight_decay=args.weight_decay,
        momentum=0.0,
        eta_min=args.eta_min,
        train_loader=train_dloader,
        device=device,
        n_epochs_per_log=args.n_epochs_per_log,
        log_function=log_function,
        loss_function=loss_module_0,
    )

    if not args.dont_write_outputs:
        #######################################################################
        # FIND EPOCH WITH MINIMAL TEST ERROR

        df = pd.read_table(args.results_fp)

        df = df[df["val_rel_l2"] == df["val_rel_l2"].min()]

        best_epoch = df.iloc[0]["epoch"]

        logging.info("Epoch %i was best; loading these weights", best_epoch)

        best_weight_fp = os.path.join(
            args.model_weights_dir, f"epoch_{best_epoch}.pickle"
        )

        #######################################################################
        # LOAD THESE MODEL WEIGHTS

        model.load_state_dict(
            torch.load(best_weight_fp, map_location=torch.device("cpu"))
        )
        model = model.to(device)

        #######################################################################
        # RUN INFERENCE ON ALL OF THE TRAIN/TEST SAMPLES

        dset_train.output_nans_bool = True
        dset_val.output_nans_bool = True

        n_train_all = len(dset_train)
        n_val_all = len(dset_val)

        train_dloader = torch.utils.data.DataLoader(
            dset_train, batch_size=args.batch_size
        )
        val_dloader = torch.utils.data.DataLoader(dset_val, batch_size=args.batch_size)
        logging.info(
            "Running inference on all %i train and %i val samples",
            n_train_all,
            n_val_all,
        )

        make_preds_on_dataset(
            model,
            train_dloader,
            device,
            args.output_dir_train,
            shard_size=500,
            experiment_info=train_metadata_dd,
        )

        make_preds_on_dataset(
            model,
            val_dloader,
            device,
            args.output_dir_val,
            shard_size=500,
            experiment_info=val_metadata_dd,
        )

    logging.info("Finished")


if __name__ == "__main__":
    a = setup_args()

    for name, logger in logging.root.manager.loggerDict.items():
        logging.getLogger(name).setLevel(logging.WARNING)

    if a.debug:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)

    id_hash = hash_dict(vars(a))
    a.hash = id_hash
    if a.dont_use_wandb:
        main(a)
    else:
        # print(vars(a))
        # print(id_hash)
        with wandb.init(
            id=id_hash,
            project=a.wandb_project_name,
            entity=a.wandb_entity,
            config=vars(a),
            mode=a.wandb_mode,
            reinit=True,
            resume=None,
            settings=wandb.Settings(start_method="fork"),
        ) as wandbrun:
            main(a)
