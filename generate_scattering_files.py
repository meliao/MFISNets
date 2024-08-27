"""
This script does the following:
1. Generates random scattering objects.
2. Saves the scattering objects to disk.
"""

import argparse
import logging
import os
import sys
from typing import Dict
import numpy as np
import wandb

from src.data.data_transformations import (
    apply_interp_2d,
    polar_to_euclidean,
    prep_conv_interp_2d,
)
from src.data.lowpass_filter import (
    prep_lpf_from_wavenum,
)

from solvers.integral_equation.data_generation_new import (
    shapes_and_non_constant_background,
)

# from src.data_io import ScatteringDataReader
from src.data.data_io import (
    save_dict_to_hdf5,
    load_hdf5_to_dict,
    save_field_to_hdf5,
    load_field_in_hdf5,
    update_field_in_hdf5,
)
from src.utils.logging_utils import hash_dict

FMT = "%(asctime)s:generate-data: %(levelname)s - %(message)s"
TIMEFMT = "%Y-%m-%d %H:%M:%S"

### Data file name constants
X_VALS = "x_vals"
RHO_VALS = "rho_vals"
THETA_VALS = "theta_vals"
SAMPLE_COMPLETION = "sample_completion"
FILE_COMPLETION = "file_completion"
SEED = "seed"
CONTRAST = "contrast"
NUM_SHAPES = "num_shapes"
GAUSSIAN_LPF_PARAM = "gaussian_lpf_param"
BACKGROUND_MAX_FREQ = "background_max_freq"
BACKGROUND_MAX_RADIUS = "background_max_radius"
Q_CART = "q_cart"
Q_POLAR = "q_polar"


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_samples", type=int)
    parser.add_argument("--n_pixels", type=int)
    parser.add_argument("--contrast", type=float)
    parser.add_argument("--output_fp", type=str)
    parser.add_argument("--background_max_radius", type=float, default=0.4)
    parser.add_argument("--background_max_freq", type=float, default=10.0)
    parser.add_argument("--spatial_domain_max", type=float, default=0.5)
    parser.add_argument("--gaussian_lpf_param", type=float, default=None)
    parser.add_argument("--n_shapes", type=int, default=3)
    parser.add_argument("--no_filtering", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_project")
    parser.add_argument(
        "--wandb_mode", choices=["offline", "online"], default="offline"
    )
    parser.add_argument("--dont_use_wandb", default=False, action="store_true")

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    ###########################################################################
    # Setup

    logging.info("Using seed: %i", args.seed)

    np.random.seed(args.seed)

    d = os.path.split(args.output_fp)[0]
    if not os.path.isdir(d):
        os.mkdir(d)

    # First, check whether the raw data file is complete already...
    # if so, we can avoid reading everything from disk
    try:
        df_already_complete = load_field_in_hdf5(
            "file_completion", args.output_fp
        ).item()
    except:
        df_already_complete = False
    logging.info(f"Data file complete? {df_already_complete}")
    # Check for file completion here
    if df_already_complete == True:
        logging.info(
            f"File marked complete; exiting early (file name: {args.output_fp})"
        )
        # File is already complete so we can go home early ^.^
        return

    ###########################################################################
    # Try Loading data file and create metadata if it's not present
    try:
        logging.info(f"Loading settings from target file")
        meas_settings = load_hdf5_to_dict(
            args.output_fp,
        )
        logging.debug(f"1. measurement settings keys: {meas_settings.keys()}")
        try:
            x_vals = meas_settings[X_VALS]
            theta_vals = meas_settings[THETA_VALS]
            rho_vals = meas_settings[RHO_VALS]
            n_rho = rho_vals.shape[0]
            if "sample_completion" in meas_settings.keys():
                # Load the sample completion status if available
                sample_completion = meas_settings["sample_completion"]
            else:
                # Otherwise, make the entry in the file and as a variable
                sample_completion = np.zeros(args.n_samples, dtype=bool)
                # Make an entry right now
                save_field_to_hdf5(
                    "sample_completion",
                    sample_completion,
                    args.output_fp,
                    overwrite=True,
                )

            if "file_completion" in meas_settings.keys():
                # Load the sample completion status if available
                file_completion = meas_settings["file_completion"]
                logging.info(f"Loaded file completion variable: {file_completion}")
            else:
                # Otherwise, make the entry in the file and as a variable
                file_completion = np.array([False])
                # Make an entry right now
                save_field_to_hdf5(
                    "file_completion",
                    file_completion,
                    args.output_fp,
                    overwrite=True,
                )

        except:
            logging.info(f"Proceeding as though no file were found")
            raise FileNotFoundError
            # treat it like a file-not-found error because we matched on
            # a file that shouldn't be there (is basically empty or defective)

        assert x_vals.shape[0] == args.n_pixels
        assert n_rho == args.n_pixels // 2

    except FileNotFoundError:
        # logging.debug(f"About to generate settings from scratch but terminating early... (debug)")
        # return

        ################################################################
        # logging.info(f"Generating settings from scratch")

        # Set up the metadata
        x_vals = np.linspace(
            -args.spatial_domain_max,
            args.spatial_domain_max,
            args.n_pixels,
            endpoint=False,
        )
        n_rho = args.n_pixels // 2
        rho_vals = np.linspace(0, args.spatial_domain_max, n_rho, endpoint=False)

        theta_vals = np.linspace(0, 2 * np.pi, args.n_pixels, endpoint=False)

        ################################################################
        # Set up the file on disk
        q_cart_dummy = np.full(
            (args.n_samples, args.n_pixels, args.n_pixels),
            np.nan,
            dtype=np.float32,
        )

        q_polar_dummy = np.full(
            (args.n_samples, args.n_pixels, n_rho),
            np.nan,
            dtype=np.float32,
        )

        sample_completion = np.zeros(args.n_samples, dtype=bool)
        file_completion = np.array([False])

        save_dict_to_hdf5(
            data_dict={
                X_VALS: x_vals,
                RHO_VALS: rho_vals,
                THETA_VALS: theta_vals,
                SEED: np.array([args.seed]),
                CONTRAST: np.array([args.contrast]),
                BACKGROUND_MAX_FREQ: np.array([args.background_max_freq]),
                BACKGROUND_MAX_RADIUS: np.array([args.background_max_radius]),
                NUM_SHAPES: np.array([args.n_shapes]),
                GAUSSIAN_LPF_PARAM: np.array([args.gaussian_lpf_param]),
                Q_CART: q_cart_dummy,
                Q_POLAR: q_polar_dummy,
                SAMPLE_COMPLETION: sample_completion,
                FILE_COMPLETION: file_completion,
            },
            fp_out=args.output_fp,
        )

        del q_cart_dummy
        del q_polar_dummy

    # Scattering object polar-to-euclidean interp objects
    polar_grid = polar_to_euclidean(theta_vals, rho_vals)  # (n_theta*n_rho, 2)
    conv_cart_to_polar_x, conv_cart_to_polar_y = prep_conv_interp_2d(
        x_vals,
        x_vals,  # Use x points for y dim here
        polar_grid,
        bc_modes=("extend", "extend"),
        a_neg_half=True,  # set a=-1/2 or -3/4 as a parameter for the conv filter
    )
    ##################################################################
    # Set up scattering objects
    dx = x_vals[1] - x_vals[0]
    if args.no_filtering:
        lpf_obj = None
        logging.info("Not using a gaussian LPF on the scattering objects.")
    else:
        lpf_obj, _, _ = prep_lpf_from_wavenum(
            args.gaussian_lpf_param,
            args.n_pixels,
        )
        logging.info(
            "Using a gaussian LPF on the scattering objects with parameter %s",
            args.gaussian_lpf_param,
        )

    n_samples = args.n_samples

    q_cart_eff = np.full(
        (n_samples, args.n_pixels, args.n_pixels), np.nan, dtype=np.float32
    )

    q_polar_eff = np.full((n_samples, args.n_pixels, n_rho), np.nan, np.float32)

    logging.info("Beginning to generate %i scattering objects", n_samples)

    for i in range(n_samples):
        q_cart_eff[i] = shapes_and_non_constant_background(
            args.n_shapes,
            args.contrast,
            args.spatial_domain_max,
            args.n_pixels,
            args.background_max_freq,
            args.background_max_radius,
            no_intersection_bool=True,
            lpf_obj=lpf_obj,
        )
        q_polar_eff[i] = apply_interp_2d(
            conv_cart_to_polar_x, conv_cart_to_polar_y, q_cart_eff[i]
        ).reshape((args.n_pixels, n_rho))

        log_dd = {
            "i": i,
            # This is a lot of output, so I don't want to log it every time.
            # "rng_state": np.random.get_state(),
        }
        if not args.dont_use_wandb:
            wandb.log(log_dd)
        else:
            logging.info(log_dd)

    full_slice = slice(0, args.n_samples)
    update_field_in_hdf5(Q_CART, q_cart_eff[full_slice], args.output_fp, full_slice)
    update_field_in_hdf5(Q_POLAR, q_polar_eff[full_slice], args.output_fp, full_slice)
    sample_completion[full_slice] = True

    # Fetch from disk again in case someone else was working on the same file at the same time
    update_field_in_hdf5(SAMPLE_COMPLETION, sample_completion, args.output_fp)

    file_completion[0] = True
    update_field_in_hdf5(
        FILE_COMPLETION,
        file_completion,
        args.output_fp,
    )
    logging.info("Finished")


if __name__ == "__main__":
    a = setup_args()

    a.seed = eval(a.seed)

    root = logging.getLogger()

    handler = logging.StreamHandler(sys.stderr)
    if a.debug:
        handler.level = logging.DEBUG
        root.setLevel(logging.DEBUG)
    else:
        handler.level = logging.INFO
        root.setLevel(logging.INFO)

    formatter = logging.Formatter(FMT, datefmt=TIMEFMT)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    hash_id = hash_dict(vars(a))

    if a.dont_use_wandb:
        main(a)
    else:
        with wandb.init(
            id=hash_id,
            project=a.wandb_project,
            entity=a.wandb_entity,
            config=vars(a),
            mode=a.wandb_mode,
            reinit=True,
            resume=None,
            settings=wandb.Settings(start_method="fork"),
        ) as wandbrun:
            main(a)
