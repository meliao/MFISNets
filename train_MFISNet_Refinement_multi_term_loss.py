"""
This script does the following:
1. Loads data from specified directories
2. Sets up an ParallelFYNetInverse model (with or without a skip connection)
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
from src.models.MFISNet_Refinement import (
    MFISNet_Refinement,
)
from src.training_utils.train_loop import train, evaluate_losses_on_dataloader
from src.training_utils.loss_functions import MultiTermLossFunction
from src.training_utils.make_predictions import make_preds_on_dataset
from src.utils.logging_utils import FMT, TIMEFMT, write_result_to_file, hash_dict


def setup_args() -> argparse.Namespace:
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
    parser.add_argument("-input_q_hat", default=False, action="store_true")
    parser.add_argument("-input_after_adj", default=False, action="store_true")
    parser.add_argument("-warm_start_init", default=False, action="store_true")
    # parser.add_argument("-lr_decrease_factor", default=None, type=float)
    parser.add_argument(
        "-loss_scale_factor",
        type=float,
        default=1.0,
        help="Values > 1 make the later freqs more important",
    )

    a = parser.parse_args()

    return a


class MultiFreqResidData(torch.utils.data.Dataset):
    def __init__(
        self,
        d_mh: torch.Tensor,
        q_polar_lpf: torch.Tensor,
        output_nan_samples_bool: bool = False,
        freq_output_idx: int = None,
    ) -> None:
        """_summary_

        Args:
            d_mh (torch.Tensor): Has shape (N_samples, N_freqs, N_m, N_h)
            q_polar_lpf (torch.Tensor): Has shape (N_samples, N_freqs, N_m, N_h)
            output_nan_samples_bool (bool, optional): Defaults to False.
        """
        self.d_mh = torch.view_as_real(d_mh).to(torch.float)
        self.q_polar_lpf = q_polar_lpf.to(torch.float)
        self.output_nan_samples_bool = output_nan_samples_bool
        logging.debug(
            "MultiFreqResidData: q_polar_lpf shape: %s", self.q_polar_lpf.shape
        )
        logging.debug("MultiFreqResidData: d_mh shape: %s", self.d_mh.shape)

        nan_samples = torch.any(torch.isnan(self.d_mh[:, :, 0, 0, 0]), dim=1)
        self.keep_indices = torch.argwhere(torch.logical_not(nan_samples)).flatten()
        logging.debug(
            "MultiFreqResidData: nan_samples has shape: %s  and sum %s",
            nan_samples.shape,
            torch.sum(nan_samples),
        )
        logging.debug(
            "MultiFreqResidData: self.keep_indices: shape: %s", self.keep_indices.shape
        )

        logging.info(
            "Initialized a MultiFreqResidData instance with d_mh shape: %s and q_polar_lpf shape: %s",
            self.d_mh.shape,
            self.q_polar_lpf.shape,
        )

        self.n_samples = self.d_mh.shape[0]
        if freq_output_idx is None:
            self.freq_output_idx = 0
        else:
            self.freq_output_idx = freq_output_idx

    def __len__(self):
        if self.output_nan_samples_bool:
            return self.n_samples
        else:
            return self.keep_indices.shape[0]

    def __getitem__(self, idx):
        """Returns (x, y, z)
        where:
         - x is the wave field d_mh at multiple different freqs. It has shape (N_M, N_H, N_freqs)
         - y is the target scattering object in polar coords. It
        """
        if not self.output_nan_samples_bool:
            idx = self.keep_indices[idx]

        x = self.d_mh[idx]
        return x, self.q_polar_lpf[idx], self.q_polar_lpf[idx, -1]

    def __repr__(self) -> str:
        _, N_freqs, N_m, N_h, _ = self.d_mh.shape
        N_samples = self.__len__()
        s = f"MultiFreqResidData object with N_m={N_m}, N_h={N_h}, N_freqs={N_freqs}, freq_output_idx={self.freq_output_idx}"
        s += f"  N_samples={N_samples}"
        return s


def load_data(
    meas_data_dir_frmt: str,
    scobj_data_dir: str,
    wavenumbers: List[str],
    truncate_num: int = None,
) -> MultiFreqResidData:
    """This function loads all of the samples indicated by the data_dir.
    It arranges the data and returns a MultiFreqResidData object.

    Args:
        data_dir (str): Where the data is located
        truncate_num (int, optional): _description_. Defaults to None.

    Returns:
        MultiFreqResidData: _description_
    """

    dd_last = load_dir(
        meas_data_dir_frmt.format(wavenumbers[-1]),
        scobj_data_dir,
        truncate_num=truncate_num,
    )

    d_mh = np.empty(
        (
            dd_last[D_MH].shape[0],
            len(wavenumbers),
            dd_last[D_MH].shape[1],
            dd_last[D_MH].shape[2],
        ),
        dtype=np.complex64,
    )
    d_mh[:, -1] = dd_last[D_MH]

    q_polar = dd_last[Q_POLAR]
    q_polar_lpf = np.empty(
        (q_polar.shape[0], len(wavenumbers), q_polar.shape[1], q_polar.shape[2])
    )
    q_polar_lpf[:, -1] = q_polar

    # logging.debug(
    #     "load_data: Now I am looping over the remaining wavenumbers: %s",
    #     wavenumbers[:-1],
    # )

    for i, w in enumerate(wavenumbers[:-1]):
        # logging.debug("load_data: w=%s, i=%s", w, i)
        dd_i = load_dir(
            meas_data_dir_frmt.format(w),
            scobj_data_dir,
            truncate_num=truncate_num,
        )
        d_mh[:, i] = dd_i[D_MH]
        q_polar_lpf[:, i] = dd_i[Q_POLAR_LPF]

    q_polar_lpf = torch.from_numpy(q_polar_lpf)
    d_mh = torch.from_numpy(d_mh)

    dset_obj = MultiFreqResidData(d_mh, q_polar_lpf, output_nan_samples_bool=False)
    metadata_dd = {i: dd_last[i] for i in KEYS_FOR_EXPERIMENT_INFO_OUT}
    return dset_obj, metadata_dd


def main(args: argparse.Namespace) -> None:
    """
    1. Load data
    2. Do necessary transformations
    3. Set up NN
    4. Train NN
    """
    # Check arguments. Some settings of the arguments may not be acceptable. In this case, we'll
    # exit early.
    if args.input_after_adj and not args.input_q_hat:
        logging.warning(
            f"Invalid argument setting: input_q_hat={args.input_q_hat} and input_after_adj={args.input_after_adj}"
        )
        return

    if args.input_q_hat and args.warm_start_init:
        logging.warning(
            f"Invalid argument setting: input_q_hat={args.input_q_hat} and warm_star_init={args.warm_start_init}"
        )
        return

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

    # Check whether this training run pre-exists
    if os.path.isfile(args.results_fp):
        pre_existing_bool = True
        results_df = pd.read_table(args.results_fp)
        last_row = results_df.iloc[-1]

        # Check to make sure all of the arguments are the same.
        assert (
            last_row["hash"] == args.hash
        ), f"Hashes don't match: {last_row['hash']} vs {args.hash}"

        final_epoch = last_row["epoch"]
        final_training_part = last_row["training_part"]
        if final_training_part.startswith("pretrain"):
            final_training_part_int = int(final_training_part.split("_")[-1])
        else:
            final_training_part_int = len(args.wavenumbers)
        logging.info(
            "Resuming after %i epochs during training part %s",
            final_epoch,
            final_training_part,
        )

    else:
        pre_existing_bool = False
        final_epoch = 0
        final_training_part = "pretrain_0"
        final_training_part_int = 0

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data from disk
    dset_train, train_metadata_dd = load_data(
        args.meas_dir_train_frmt,
        args.scobj_dir_train,
        args.wavenumbers,
        args.truncate_num,
    )
    dset_val, val_metadata_dd = load_data(
        args.meas_dir_val_frmt,
        args.scobj_dir_val,
        args.wavenumbers,
        args.truncate_num_val,
    )

    # Grab some metadata
    x_vals = train_metadata_dd[X_VALS]
    rho_vals = train_metadata_dd[RHO_VALS]
    theta_vals = train_metadata_dd[THETA_VALS]
    N_x = x_vals.shape[0]
    N_rho = rho_vals.shape[0]
    N_theta = theta_vals.shape[0]
    N_freqs = len(args.wavenumbers)
    logging.info("Working with %i wavenumbers: %s", N_freqs, args.wavenumbers)

    n_train = len(dset_train)
    n_val = len(dset_val)

    args.n_train = n_train
    args.n_val = n_val

    logging.info("Done Data Loading. N_train=%i, N_val=%i", n_train, n_val)

    #########################################################
    # Set dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training on device: %s", device)

    #########################################################
    # Set up NN architecture

    model = MFISNet_Refinement(
        N_h=N_rho,
        N_rho=N_rho,
        c_1d=args.n_cnn_channels_1d,
        c_2d=args.n_cnn_channels_2d,
        w_1d=args.kernel_size_1d,
        w_2d=args.kernel_size_2d,
        N_cnn_1d=args.n_cnn_1d,
        N_cnn_2d=args.n_cnn_2d,
        N_freqs=len(args.wavenumbers),
        return_all_q_hats=True,
    )

    if pre_existing_bool:
        fp_weights = os.path.join(args.model_weights_dir, f"epoch_{final_epoch}.pickle")
        logging.info("Loading weights from file: %s", fp_weights)
        incompatible_keys = model.load_state_dict(
            torch.load(fp_weights, map_location="cpu")
        )
        logging.info("Message from model loading: %s", incompatible_keys)

    ###########################################################################
    # SET UP LOG FUNCTION
    def log_function(model_0, epoch_local):
        """
        NEED TO SET ANEW:
        loss_fn_dd
        epoch_stagger
        training_part
        n_epochs_this_step

        OTHER OUTER SCOPE STUFF:
        train_dloader,
        val_dloader,
        device,
        args
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
                "%s Epoch %i/%i. Train MSE: %f, Train Rel L2: %f, Train Loss Val: %f",
                training_part,
                epoch_local,
                n_epochs_this_step,
                torch.mean(train_loss_dd["mse"]).item(),
                torch.mean(train_loss_dd["rel_l2"]).item(),
                torch.mean(train_loss_dd["loss_value"]).item(),
                # train_rel_l2_aaa,
                # train_psnr_aaa,
            )
            logging.info(
                "\t Val MSE: %f, Val Rel L2: %f, Val Loss Val: %f",
                torch.mean(val_loss_dd["mse"]).item(),
                torch.mean(val_loss_dd["rel_l2"]).item(),
                torch.mean(val_loss_dd["loss_value"]).item(),
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
                "hash": args.hash,
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

    ##########################################################################################
    # PRETRAIN EACH BLOCK
    loss_module = MultiTermLossFunction(
        pred_idx=N_freqs - 1, scale_factor=args.loss_scale_factor
    )
    loss_fn_dd = {
        "mse": loss_module.mse,
        "psnr": loss_module.psnr,
        "rel_l2": loss_module.relative_l2_error,
        "final_mse": loss_module.mse_against_final,
        "final_psnr": loss_module.psnr_against_final,
        "final_rel_l2": loss_module.relative_l2_error_against_final,
        "loss_value": loss_module,
    }
    train_dloader = torch.utils.data.DataLoader(dset_train, batch_size=args.batch_size)
    val_dloader = torch.utils.data.DataLoader(dset_val, batch_size=args.batch_size)
    training_part = "fine-tune"
    epoch_stagger = 0
    n_epochs_this_step = args.n_epochs
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
        loss_function=loss_module,
    )

    if not args.dont_write_outputs:
        #######################################################################
        # FIND EPOCH WITH MINIMAL TEST ERROR

        df = pd.read_table(args.results_fp)
        df_this_part = df[df["training_part"] == training_part]

        df_best = df_this_part[
            df_this_part["val_rel_l2"] == df_this_part["val_rel_l2"].min()
        ]

        best_epoch = df_best.iloc[0]["epoch"]

        logging.info("Epoch %i was best; loading these weights", best_epoch)

        best_weight_fp_final = os.path.join(
            args.model_weights_dir, f"epoch_{best_epoch}.pickle"
        )
        model.load_state_dict(
            torch.load(best_weight_fp_final, map_location=torch.device("cpu"))
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

        model.return_all_q_hats = False
        model.freq_output_idx = N_freqs - 1

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
