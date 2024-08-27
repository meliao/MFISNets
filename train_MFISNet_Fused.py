# Train the MFISNet-Fused variant

import logging
from typing import List
import argparse
from timeit import default_timer
import os
import numpy as np
import torch
import wandb
import os, psutil  # to fetch memory usage

from src.data.add_noise import add_noise_to_d

from src.data.data_io import (
    load_dir,
)
from src.models.MFISNet_Fused import MFISNet_Fused
from src.training_utils.train_loop import train, evaluate_losses_on_dataloader
from src.training_utils.loss_functions import (
    MSEModule,
)
from src.utils.logging_utils import FMT, TIMEFMT, write_result_to_file, hash_dict

from src.data.data_naming_constants import (
    Q_POLAR,
    Q_CART,
    D_MH,
    D_RS,
    Q_POLAR_LPF,
    Q_CART_LPF,
    NU_SF,
    OMEGA_SF,
    KEYS_FOR_TRAINING_SAMPLES_ALL,
)

FREQ_DEPENDENT_KEYS = [
    D_MH,
    D_RS,
    Q_CART_LPF,
    Q_POLAR_LPF,
    NU_SF,
    OMEGA_SF,
]
TRUNCATABLE_KEYS = [
    Q_POLAR,
    Q_CART,
    D_MH,
    D_RS,
    Q_POLAR_LPF,
    Q_CART_LPF,
]


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir_base",
        type=str,
        help="Indicate the directory containing all the measurement folders"
        " corresponding to the relevant frequencies and data subsets",
    )
    parser.add_argument("--data_input_nus", type=str, nargs="+")
    parser.add_argument("--eval_on_test_set", default=False, action="store_true")
    parser.add_argument(
        "--no_eval_on_test_set", action="store_false", dest="eval_on_test_set"
    )
    parser.add_argument("--train_results_fp")
    parser.add_argument("--model_weights_dir")
    parser.add_argument("--truncate_num", type=int)
    parser.add_argument("--truncate_num_val", type=int)
    parser.add_argument("--seed", type=int, default=35675)
    parser.add_argument("--n_cnn_1d", type=int, default=3)
    parser.add_argument("--n_cnn_2d", type=int, default=3)
    parser.add_argument("--n_cnn_channels_1d", type=int, default=10)
    parser.add_argument("--n_cnn_channels_2d", type=int, default=10)
    parser.add_argument("--kernel_size_1d", type=int, default=13)
    parser.add_argument("--kernel_size_2d", type=int, default=13)
    parser.add_argument("--merge_middle_freq_channels", type=str)
    # parser.add_argument("--merge_middle_freq_channels", action="store_true",
    #                     dest="merge_middle_freq_channels_bool")
    # parser.add_argument("--no_merge_middle_freq_channels", action="store_false",
    #                     dest="merge_middle_freq_channels_bool")
    parser.add_argument("--polar_padding", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=100)
    # parser.add_argument("--n_epochs_pretrain_0", type=int, default=10)
    # parser.add_argument("--n_epochs_pretrain_1", type=int, default=10)
    parser.add_argument("--lr_init", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eta_min", type=float, default=1e-04)
    parser.add_argument("--n_epochs_per_log", type=int, default=5)
    # parser.add_argument("--omega_0_idx", type=int, default=1)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--forward_model_adjustment", type=float, default=1.0)
    parser.add_argument(
        "--noise_to_signal_ratio", default=None, type=float
    )  # train and test with noise

    # Weights and Biases setup
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, help="The W&B entity")
    parser.add_argument(
        "--wandb_mode", choices=["offline", "online", "disabled"], default="offline"
    )

    # Misc. options
    # parser.add_argument("--concat_wave_fields", default=False, action="store_true")
    # parser.add_argument("--skip_connections", default=False, action="store_true")
    # parser.add_argument("--forward_network", default=False, action="store_true")
    parser.add_argument("--big_init", default=False, action="store_true")
    parser.add_argument("--small_init", action="store_false", dest="big_init")
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
        # (OOT 6/4/2024) Gives two copies of the target because the training
        # loop seems to expect a filtered and final version of the sample
        return self.X[idx], self.y[idx], self.y[idx]


def load_multifreq_dataset(
    freq_dir_list: List[str],
    # concat_wave_fields: bool = False,
    # no_filtering_bool: bool = False,
    truncate_num: int = None,
    key_replacement: dict = None,
    noise_to_sig_ratio: float = None,
    add_noise_to: str = None,
    load_cart: bool = False,
    nan_mode: str = None,
) -> dict:
    """
    Helper function to load datasets containing multiple frequencies
    Allows for replacing keys to ensure the right naming convention

    Parameters:
        freq_dir_list (List of str): give the different directories corresponding to the different frequencies
        truncate_num (int): the number of samples to be loaded
        key_replacement (dict): key mapping in case old field names need to be overriden
            Note: should not be needed but is left as a courtesy to outdated code
        noise_to_sig_ratio (float): level of noise relative to the signal
        add_noise_to (str): specify whether to add noise to "d_mh" or "d_rs".
            Only adds to one of these because the noise patterns will be different on each
        nan_mode (str): choose between "zero" out nan entries or "skip" entire samples containing a nan

    Outputs:
        dd (dict): dictionary representing the dataset
    """
    N_freqs = len(freq_dir_list)
    dd_list = []
    for dir_name in freq_dir_list:
        logging.info(f"Loading dataset from {dir_name}")
        dd_new = load_dir(
            dir_name,  # pass as scobj dir
            dir_name,  # pass as meas  dir
            truncate_num=truncate_num,
            # load_all_fields=True,
            # new_naming_mode=True,
            # key_replacement=key_replacement,
            load_cart_and_rs=load_cart,
        )
        dd_list.append(dd_new)

    # Set up the dictionary for fixed values and fields that will get multiple frequencies
    dd_all = dict()
    simple_fdk_list = []
    present_fdk_list = []
    for key, val in dd_list[0].items():
        if key not in FREQ_DEPENDENT_KEYS:  # or key in [OMEGA_SF, NU_SF]:
            dd_all[key] = val
        elif key in {OMEGA_SF, NU_SF}:
            dd_all[key] = np.zeros(N_freqs)
            simple_fdk_list.append(key)
        else:
            # Assume we always just put the new frequency index in dim 1
            curr_shape = dd_list[0][key].shape
            logging.info(f"Found fdk {key} whose entry has shape {curr_shape}")
            new_shape = tuple([*curr_shape[:1], N_freqs, *curr_shape[1:]])
            dd_all[key] = np.zeros(new_shape, dtype=dd_list[0][key].dtype)
            present_fdk_list.append(key)

    for fdk in present_fdk_list:
        for fi in range(N_freqs):
            logging.info(
                f"(key {fdk}) Loading value of shape {dd_list[fi][fdk].shape} into a slice"
                f" of {dd_all[fdk].shape}"
            )
            dd_all[fdk][:, fi] = dd_list[fi][fdk]
    for simple_fdk in simple_fdk_list:
        for fi in range(N_freqs):
            dd_all[simple_fdk][fi] = dd_list[fi][simple_fdk].item()

    # Deal with NaNs; determine using d_mh
    nan_mode = nan_mode.lower() if nan_mode is not None else "skip"
    if nan_mode == "keep":
        # Keep the NaNs
        for key in TRUNCATABLE_KEYS:
            if key in dd_all.keys() and key in KEYS_FOR_TRAINING_SAMPLES_ALL:
                dd_all[key] = np.nan_to_num(dd_all[key], copy=False, nan=np.nan)
    if nan_mode == "zero":
        # Zero out for everything
        for key in TRUNCATABLE_KEYS:
            if key in dd_all.keys() and key in KEYS_FOR_TRAINING_SAMPLES_ALL:
                dd_all[key] = np.nan_to_num(dd_all[key], copy=False, nan=0)
    elif nan_mode == "skip":
        # wave_field_mh shape: (N_samples, N_freqs, N_m, N_h)
        keep_idcs = np.logical_not(np.any(np.isnan(dd_all[D_MH][:, :, 0, 0]), axis=1))
        for key in TRUNCATABLE_KEYS:
            if key in dd_all.keys() and key in KEYS_FOR_TRAINING_SAMPLES_ALL:
                dd_all[key] = dd_all[key][keep_idcs]

    # Add noise if applicable
    if noise_to_sig_ratio is not None:
        add_noise_to = add_noise_to.lower() if add_noise_to is not None else "d_mh"
        if add_noise_to == "d_mh":
            dd_all[D_MH] = add_noise_to_d(dd_all[D_MH], noise_to_sig_ratio)
        elif add_noise_to == "d_rs":
            dd_all[D_RS] = add_noise_to_d(dd_all[D_RS], noise_to_sig_ratio)
        else:
            raise ValueError(
                f"Did not recognize {add_noise_to} as a valid field to add noise to."
                f" Please enter either 'd_mh' or 'd_rs'."
            )
        logging.info(
            f"Applied noise at {noise_to_sig_ratio:.2f} to field '{add_noise_to}'"
        )

    # Apply key replacement
    key_replacement = key_replacement if key_replacement is not None else {}

    # Replace one key at a time to reduce memory overhead...hopefully...
    for old_key in key_replacement.keys():
        if old_key not in dd_all.keys():
            continue  # skip if key is invalid
        new_key = key_replacement[key]
        if new_key == old_key:
            continue  # skip if no move is required
        dd_all[new_key] = dd_all[old_key]
        del dd_all[old_key]

    return dd_all


def setup_single_dataset(q_polar: np.ndarray, wave_field_mh: np.ndarray) -> LinearData:
    """
    Set up a single (multi-frequency) dataset (such as training/eval)
    Parameters:
        # data_dd (dict): dictionary received while loading the dataset
        q_polar (np.ndarray): stack of scattering objects
        wave_field_mh (np.ndarray): stack of wavefield patterns
    Return values:
        dset (LinearData): torch-ready data
    """
    inputs_dset = torch.view_as_real(torch.from_numpy(wave_field_mh))
    targets_dset = torch.from_numpy(q_polar)
    dset = LinearData(inputs_dset, targets_dset)
    return dset


def main(
    args: argparse.Namespace,
    # Extra arguments for testing purposes
    # skip_wandb: bool = False,
    return_model: bool = False,
) -> None:
    """
    1. Load data
    2. Do necessary transformations
    3. Set up NN
    4. Train NN
    """
    mmfc_bool = False if args.merge_middle_freq_channels.lower() == "false" else True
    polar_padding_bool = False if args.polar_padding.lower() == "false" else True
    logging.info(
        f"Received: merge_middle_freq_channels={mmfc_bool} and polar_pad={polar_padding_bool}"
    )
    args.merge_middle_freq_channels_bool = mmfc_bool
    args.polar_padding_bool = polar_padding_bool

    if not os.path.isdir(args.model_weights_dir):
        os.mkdir(args.model_weights_dir)

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    # os.path.join()
    train_files = [
        os.path.join(data_dir_base, f"train_measurements_nu_{nu}") for nu in str_nu_list
    ]
    val_files = [
        os.path.join(data_dir_base, f"val_measurements_nu_{nu}") for nu in str_nu_list
    ]
    if args.eval_on_test_set:
        test_files = [
            os.path.join(data_dir_base, f"test_measurements_nu_{nu}")
            for nu in str_nu_list
        ]
    else:
        test_files = None

    logging.info(
        f"Attempting to load the following folders: {train_files} and {val_files}"
    )

    # Training data dictionary
    # key_replacement = {
    #     "Theta_vals": "theta_vals",
    #     "Input": "q_polar",
    #     # "Input_Cart": "q_cart",
    # }
    logging.info(f"Loading training dataset")
    train_dd = load_multifreq_dataset(
        train_files,
        truncate_num=args.truncate_num,
        # key_replacement=key_replacement,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_mh",
        nan_mode="skip",
        load_cart=False,
    )

    def kv_shrinker(key, val):
        """Little helper function to see the shapes of entires in a dictionary"""
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
    logging.info(f"train_dd has entries with shapes: {train_dd_short}")

    # Evaluation data dictionary
    logging.info(f"Loading evaluation dataset")
    eval_dd = load_multifreq_dataset(
        test_files if args.eval_on_test_set else val_files,
        truncate_num=args.truncate_num_val,
        # key_replacement=key_replacement,
        noise_to_sig_ratio=args.noise_to_signal_ratio,
        add_noise_to="d_mh",
        nan_mode="skip",
        load_cart=False,
    )
    eval_dd_short = dict(kv_shrinker(k, v) for (k, v) in eval_dd.items())
    logging.info(f"eval_dd has entries with shapes: {eval_dd_short}")

    # logging.info(f"Received a dictionary with keys: {list(train_dd.keys())}")
    train_q_polar = train_dd["q_polar"]
    # train_q_cart  = train_dd["q_cart"]
    train_d_mh = train_dd["d_mh"]

    rho_vals = train_dd["rho_vals"]
    theta_vals = train_dd["theta_vals"]
    h_vals = train_dd["h_vals"]
    omega_vals = train_dd["omega_sf"]
    x_vals = train_dd["x_vals"]

    N_rho = rho_vals.shape[0]
    N_h = h_vals.shape[0]
    N_theta = theta_vals.shape[0]
    N_train = train_q_polar.shape[0]
    N_eval = eval_dd["q_polar"].shape[0]

    # Next... run the "setup_dataset" function (may require re-organizing)
    train_dset = setup_single_dataset(train_q_polar, train_d_mh)
    eval_dset = setup_single_dataset(eval_dd["q_polar"], eval_dd["d_mh"])
    logging.info(f"Finished loading data. N_train={N_train}, N_eval={N_eval}")

    ### Prepare for NN training ###
    # Set up CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Training on device: %s", device)
    # Send to the data loader
    train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size)
    eval_dloader = torch.utils.data.DataLoader(eval_dset, batch_size=args.batch_size)

    # Initialize the model
    model = MFISNet_Fused(
        N_h=N_h,
        N_rho=N_rho,
        N_freqs=N_freqs,
        c_1d=args.n_cnn_channels_1d,
        c_2d=args.n_cnn_channels_2d,
        w_1d=args.kernel_size_1d,
        w_2d=args.kernel_size_2d,
        N_cnn_1d=args.n_cnn_1d,
        N_cnn_2d=args.n_cnn_2d,
        merge_middle_freq_channels=args.merge_middle_freq_channels_bool,
        big_init=args.big_init,
        polar_padding=args.polar_padding_bool,
    )

    ########################### Training procedure ###########################
    N_epochs = args.n_epochs

    # loss_module_0 = MSEModule(loss_idx=slice(None), final_output_idx=slice(None))
    loss_module_0 = MSEModule()
    loss_fn_dd = {
        "mse": loss_module_0.mse,
        "psnr": loss_module_0.psnr,
        "rel_l2": loss_module_0.relative_l2_error,
        "final_mse": loss_module_0.mse_against_final,
        "final_psnr": loss_module_0.psnr_against_final,
        "final_rel_l2": loss_module_0.relative_l2_error_against_final,
    }

    id_hash = hash_dict(vars(args))
    epoch_stagger = 0  # Just a single training step

    # Spin up the Weights and Biases environment
    with wandb.init(
        id=id_hash,
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        # mode="disabled" if skip_wandb else args.wandb_mode,
        mode=args.wandb_mode,
        reinit=True,
        resume=None,
        settings=wandb.Settings(start_method="fork"),
    ) as wandbrun:
        # First, set up the logging function
        def log_function(model_0, epoch_local):
            """
            Need to set:
            - loss_fn_dd
            """
            with torch.no_grad():
                epoch_eff = epoch_stagger + epoch_local

                # 1. Evaluate on train set
                train_loss_dd = evaluate_losses_on_dataloader(
                    model_0, train_dloader, loss_fn_dd, device
                )
                eval_loss_dd = evaluate_losses_on_dataloader(
                    model_0, eval_dloader, loss_fn_dd, device
                )

                weight_norm = torch.norm(
                    torch.cat([x.view(-1) for x in model_0.parameters()]), 2
                )

                # 3. Log to console and log file
                logging.info(
                    "Epoch %i/%i. Train MSE: %f, Train Rel L2: %f, Train PSNR: %f",
                    epoch_local,
                    N_epochs,
                    torch.mean(train_loss_dd["mse"]).item(),
                    torch.mean(train_loss_dd["rel_l2"]).item(),
                    torch.mean(train_loss_dd["psnr"]).item(),
                    # train_rel_l2_aaa,
                    # train_psnr_aaa,
                )
                logging.info(
                    "\t Val MSE: %f, Val Rel L2: %f, Val PSNR: %f",
                    torch.mean(eval_loss_dd["mse"]).item(),
                    torch.mean(eval_loss_dd["rel_l2"]).item(),
                    torch.mean(eval_loss_dd["psnr"]).item(),
                    # test_mse_aaa,
                    # test_rel_l2_aaa,
                    # test_psnr_aaa,
                )
                logging.info("\t Weight L2 norm: %f", weight_norm.item())

                process = psutil.Process()
                logging.info(
                    f"Memory usage: {process.memory_info().rss>>20} MB"
                )  # this is not where the memory usage peaks
                if torch.cuda.is_available():
                    vram_free_bytes, vram_available_bytes = torch.cuda.mem_get_info()
                    vram_used_mb = (vram_available_bytes - vram_free_bytes) >> 20
                    logging.info(
                        f"Current VRAM usage: {vram_used_mb} MB / {vram_available_bytes>>20} MB"
                    )

                train_dd = {
                    # Optimization info
                    "epoch": epoch_local + epoch_stagger,
                    "weight_norm": weight_norm.item(),
                    # Experiment info
                    "eval_on_val_set": not args.eval_on_test_set,
                    "eval_on_test_set": args.eval_on_test_set,
                    "n_train": N_train,
                    "n_eval": N_eval,
                    "n_freqs": N_freqs,
                    "n_cnn_1d": args.n_cnn_1d,
                    "n_cnn_2d": args.n_cnn_2d,
                    "n_cnn_channels_1d": args.n_cnn_channels_1d,
                    "n_cnn_channels_2d": args.n_cnn_channels_2d,
                    "merge_middle_freq_channels": args.merge_middle_freq_channels_bool,
                    "polar_padding": args.polar_padding_bool,
                    "kernel_size_1d": args.kernel_size_1d,
                    "kernel_size_2d": args.kernel_size_2d,
                    "lr_init": args.lr_init,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "big_init": args.big_init,
                    "eta_min": args.eta_min,
                    "n_rho_vals": N_rho,
                    "n_theta_vals": N_theta,
                    # "omega_0_idx": args.omega_0_idx,
                    # "skip_connections": args.skip_connections,
                    # "forward_network": args.forward_network,
                    "hash": id_hash,
                    # Extra data
                    "source_nu_list": nu_list,
                }
                for k, v in train_loss_dd.items():
                    train_dd["train_" + k] = torch.mean(v).item()
                for k, v in eval_loss_dd.items():
                    train_dd["eval_" + k] = torch.mean(v).item()
                write_result_to_file(args.train_results_fp, **train_dd)

                # Try to log results to W&B
                try:
                    wandbrun.log(train_dd)
                except ValueError:
                    logging.error("Error: wandb logging failed for %s" % wandbrun.id)

            fp_weights = os.path.join(
                args.model_weights_dir, f"epoch_{epoch_eff}.pickle"
            )
            torch.save(model_0.state_dict(), fp_weights)
            model_0 = model_0.to(device)

        for p in model.parameters():
            logging.info(
                f"Parameter with shape {p.shape} requires grad: {p.requires_grad}"
            )

        # Now train it!
        model = train(
            model=model,
            n_epochs=N_epochs,
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
    logging.info("Finished!")
    if return_model:
        return model
    return


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
