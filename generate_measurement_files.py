"""
This script does the following:
1. Grabs the scattering object file
2. Generates wave fields for the scattering objects.
5. Makes the necessary tranformations from FY19.
5. Saves the data to disk.
"""

import argparse
import logging
import os
import sys
from typing import Dict
import numpy as np
import wandb
from solvers.integral_equation.HelmholtzSolver import (
    setup_accelerated_solver,
)
from src.data.data_transformations import (
    prep_rs_to_mh_interp,
    apply_interp_2d,
    get_scale_factor,
    CONST_RHO_PRIME,
    CONST_THETA_PRIME,
    polar_to_euclidean,
    prep_conv_interp_2d,
)
from src.data.lowpass_filter import (
    prep_lpf_from_wavenum,
    apply_filter_fourier_2d,
)


from src.data.data_io import (
    save_dict_to_hdf5,
    load_hdf5_to_dict,
    load_field_in_hdf5,
    update_field_in_hdf5,
)
from src.utils.logging_utils import hash_dict
from torch._C import _LinAlgError
import torch.cuda
import psutil

FMT = "%(asctime)s:generate-data: %(levelname)s - %(message)s"
TIMEFMT = "%Y-%m-%d %H:%M:%S"


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_fp", type=str)
    parser.add_argument("--output_fp", type=str)
    parser.add_argument("--nu_source_freq", type=float)  # non-angular
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--receiver_radius", type=float, default=100.0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument(
        "--create_n_samples", type=int, default=-1
    )  # by default do everything
    parser.add_argument("--write_every_n", type=int, default=1)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--no_filtering", default=False, action="store_true")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_project")
    parser.add_argument(
        "--wandb_mode", choices=["offline", "online"], default="offline"
    )
    parser.add_argument("--dont_use_wandb", default=False, action="store_true")

    return parser.parse_args()


def create_new_meas_data_file(sdf_fp: str, mdf_fp: str, args: dict) -> None:
    """Creates a blank file with the desired settings and sets up empty fields
    (in case the target output does not yet exist)
    Parameters:
        sdf_fp (str): scattering object datafile filepath
        mdf_fp (str): measurement datafile filepath
        args (dict): the rest of the arguments passed to this script

    Output:
        None (simply creates the new file at sdf_fp and will raise an error if it exists already)
    """
    scobj_settings = load_hdf5_to_dict(sdf_fp)

    ### Set up new fields ###
    # Grid points
    rho_vals = scobj_settings["rho_vals"]
    theta_vals = scobj_settings["theta_vals"]
    # x_vals = scobj_settings["x_vals"]

    num_rho = rho_vals.shape[0]
    num_theta = theta_vals.shape[0]
    num_r = num_theta
    num_s = num_theta
    num_m = num_theta
    num_h = num_rho

    m_vals = theta_vals
    h_vals = np.linspace(-np.pi / 2, np.pi / 2, num_rho, endpoint=False)

    # Scattering object
    q_cart = scobj_settings["q_cart"]
    q_polar = scobj_settings["q_polar"]
    num_samples = q_cart.shape[0]
    q_cart_lpf = np.full(q_cart.shape, np.nan, dtype=np.float32)
    q_polar_lpf = np.full(q_polar.shape, np.nan, dtype=np.float32)

    # Measured wavefields
    nu_sf = np.array([args.nu_source_freq])
    omega_sf = 2 * np.pi * nu_sf
    d_rs = np.full((num_samples, num_r, num_s), np.nan, dtype=np.complex64)
    d_mh = np.full((num_samples, num_m, num_h), np.nan, dtype=np.complex64)

    # Convenience settings
    sample_completion = np.zeros(num_samples, dtype=bool)
    file_completion = np.array([False])

    new_settings = {
        # Grid points
        "m_vals": m_vals,
        "h_vals": h_vals,
        # Scattering objects
        "q_cart_lpf": q_cart_lpf,
        "q_polar_lpf": q_polar_lpf,
        # Measured wavefields
        "nu_sf": nu_sf,
        "omega_sf": omega_sf,
        "d_rs": d_rs,
        "d_mh": d_mh,
        # Convenience settings
        "sample_completion": sample_completion,
        "file_completion": file_completion,
    }

    # Combine and overwrite settings as necessary
    mdf_settings = {**scobj_settings, **new_settings}
    save_dict_to_hdf5(mdf_settings, mdf_fp)
    return


def main(args: argparse.Namespace) -> None:
    ###########################################################################
    # Setup

    # print(f"start index: {args.start_idx}")

    d = os.path.split(args.output_fp)[0]
    if not os.path.isdir(d):
        os.mkdir(d)

    ### 1. Check whether the target measurement file is complete already
    # if so, we can avoid reading everything from disk
    try:
        mdf_already_complete = load_field_in_hdf5(
            "file_completion", args.output_fp
        ).item()
    except:
        # Mark incomplete if the mdf doesn't exist
        # or has an issue with its "file_completion" field
        mdf_already_complete = False
    # logging.warning(f"Measurement data file complete? {mdf_already_complete}")
    # Check for file completion here
    if mdf_already_complete == True:
        logging.warning(
            f"Measurement file marked complete; exiting early (file name: {args.output_fp})"
        )
        return

    # May raise a FileNotFoundError if the scattering file does not exist
    invalid_sdf = False
    if not os.path.exists(args.input_fp):
        invalid_sdf = True
    else:
        # Check the "file_completion" flag
        sdf_already_complete = load_field_in_hdf5(
            "file_completion", args.input_fp
        ).item()
        invalid_sdf = not sdf_already_complete
    if invalid_sdf:
        logging.error(
            f"Expected to find a valid input scattering object file at"
            f" {args.input_fp} (and did not)"
        )
        raise FileNotFoundError(
            f"Expected to find a valid input scattering object file at"
            f" {args.input_fp} (and did not)"
        )
    else:
        num_samples_all = load_field_in_hdf5("sample_completion", args.input_fp).shape[
            0
        ]

    ### 2. Attempt to load the output measurement file if it exists
    # If no measurement file exists, create a new one
    try:
        logging.warning(f"Attempting to load settings from target file")
        # print(f"meas file exists: {os.path.exists(args.output_fp)}")
        if not os.path.exists(args.output_fp):
            raise FileNotFoundError
        meas_settings = load_hdf5_to_dict(args.output_fp)
        logging.debug("meas_settings: %s", meas_settings.keys())
    except Exception as e:
        # In case of an error while loading
        if os.path.exists(args.output_fp):
            logging.warning(
                f"Deleting measurement output file {args.output_fp} after"
                " encountering an error {e} while attempting to load output file"
            )
            os.remove(args.output_fp)
        logging.warning(f"Creating new measurement file from scratch")
        create_new_meas_data_file(args.input_fp, args.output_fp, args)
        meas_settings = load_hdf5_to_dict(args.output_fp)

    # Unload the settings into local variables
    # Grid variables
    # omega_wf = args.omega_val # wave field measurement omega
    nu_sf = args.nu_source_freq  # non-angular frequency of the source wave
    omega_sf = 2 * np.pi * nu_sf  # angular frequency of the source wave
    x_vals = meas_settings["x_vals"]
    rho_vals = meas_settings["rho_vals"]
    theta_vals = meas_settings["theta_vals"]
    m_vals = meas_settings["m_vals"]
    h_vals = meas_settings["h_vals"]
    num_rho = rho_vals.shape[0]
    num_theta = theta_vals.shape[0]
    num_x = x_vals.shape[0]
    num_pixels = x_vals.shape[0]
    num_r = num_theta
    num_s = num_theta
    num_m = num_theta
    num_h = num_rho
    # Don't load the entire q/d object and instead load just the effective chunk later

    sample_completion = meas_settings["sample_completion"]
    # num_samples_all = _settings["sample_completion"].shape[0] # already calculated above

    ### 3. set up the PDE solver and convolution operators
    logging.warning("Setting up solvers and convolution objects")
    spatial_domain_max = np.max(np.abs(x_vals))
    solver_obj = setup_accelerated_solver(
        num_pixels, spatial_domain_max, nu_sf, args.receiver_radius
    )
    num_theta = theta_vals.shape[0]
    num_h = h_vals.shape[0]
    # Measurement change-of-coordinates interp objects
    conv_rs_to_m, conv_rs_to_h = prep_rs_to_mh_interp(
        theta_vals,  # r grid points
        theta_vals,  # s grid points
        num_theta,
        num_h,
        a_neg_half=True,
    )
    # Scattering object polar-to-euclidean interp objects
    polar_grid = polar_to_euclidean(theta_vals, rho_vals)  # (n_theta*n_rho, 2)
    conv_cart_to_polar_x, conv_cart_to_polar_y = prep_conv_interp_2d(
        x_vals,
        x_vals,  # Use x points for y dim here
        polar_grid,
        bc_modes=("extend", "extend"),
        a_neg_half=True,  # set a=-1/2 or -3/4 as a parameter for the conv filter
    )

    # LPF object to make q_cart_lpf and q_polar_lpf
    dx = x_vals[1] - x_vals[0]
    nu_lpf = 2 * nu_sf
    lpf_x, _, _ = prep_lpf_from_wavenum(nu_lpf, num_x, pad_mode="power-of-two")
    lpf_y = np.copy(lpf_x)  # just reuse since x_vals=y_vals

    logging.warning(f"Finished setting up solver objects and conv operators")

    ### 4. Prepare the index range
    # num_samples_all = args.total_n_samples
    create_n_samples = (
        args.create_n_samples if args.create_n_samples != -1 else num_samples_all
    )
    args.end_idx = min(args.start_idx + create_n_samples, num_samples_all)
    full_slice = slice(args.start_idx, args.end_idx)
    num_samples_eff = args.end_idx - args.start_idx

    # Buffer variables
    # read in q/d from the measurement file
    q_cart_eff = meas_settings["q_cart"][full_slice]
    # q_polar_eff = meas_settings["q_polar"][full_slice]
    q_cart_lpf_eff = meas_settings["q_cart_lpf"][full_slice]
    q_polar_lpf_eff = meas_settings["q_polar_lpf"][full_slice]
    d_rs_eff = meas_settings["d_rs"][full_slice]
    d_mh_eff = meas_settings["d_mh"][full_slice]

    ### 5. Filter and scatter the inputs
    logging.warning(
        f"Beginning to process (filter+scatter) {num_samples_eff} scattering objects"
    )

    # chunk_counter = 0 # absolute index from the beginning
    for chunk_start_idx in range(args.start_idx, args.end_idx, args.write_every_n):
        chunk_end_idx = min(chunk_start_idx + args.write_every_n, args.end_idx)

        # Loop over the indices in the chunk
        for idx_abs in range(chunk_start_idx, chunk_end_idx):
            # idx_abs = i # rename for clarity...
            idx_eff = idx_abs - args.start_idx
            logging.warning("Working on sample %i of %i", idx_eff + 1, num_samples_eff)
            computed_soln_bool = None  # Leave blank for now...

            # It's possible the wave fields for this sample has already been computed, so we
            # want to skip computing it if possible
            is_any_scobj_nans = np.any(np.isnan(q_cart_lpf_eff[idx_eff])) or np.any(
                np.isnan(q_polar_lpf_eff[idx_eff])
            )
            # (Re-)do the filtering if needed
            if is_any_scobj_nans:
                # Redo the filtering
                q_cart_lpf_i = apply_filter_fourier_2d(
                    q_cart_eff[idx_eff],
                    lpf_x,
                    lpf_y,
                )
                q_cart_lpf_eff[idx_eff] = q_cart_lpf_i
                q_polar_lpf_i = apply_interp_2d(
                    conv_cart_to_polar_x,
                    conv_cart_to_polar_y,
                    q_cart_lpf_i,
                ).reshape(num_theta, num_rho)
                q_polar_lpf_eff[idx_eff] = q_polar_lpf_i

            # (Re-)do the PDE solve if necessary
            # First determine whether that is necessary
            is_any_data_nans = np.any(np.isnan(d_rs_eff[idx_eff])) and np.all(
                np.isnan(d_mh_eff[idx_eff])
            )
            is_all_data_nans = np.all(np.isnan(d_rs_eff[idx_eff])) and np.all(
                np.isnan(d_mh_eff[idx_eff])
            )
            # logging.warning(f"Current entry has NaNs?    {is_any_data_nans}")
            # logging.warning(f"Current entry is all NaNs? {is_all_data_nans}")
            if not is_any_data_nans:
                sample_completion[idx_abs] = True
                logging.warning(
                    f"Identifying an existing solution at index {idx_eff} from lack of NaNs"
                )
            already_present_bool = sample_completion[idx_abs]
            if already_present_bool:
                logging.warning("Solution at index %i is already present", idx_eff)
            else:
                # Now run the PDE solver in batches
                scattering_obj_i = q_cart_eff[idx_eff]
                try:
                    # This for loop calls the PDE solver, breaking the incident wave
                    # directions into batches of size (batch_size). Sometimes there's a
                    # singular matrix, which is why we have the error catching
                    for j in range(0, num_pixels, args.batch_size):
                        j_upper = min(j + args.batch_size, num_pixels)
                        directions = solver_obj.source_dirs[j:j_upper]
                        u_scat_ext = solver_obj.Helmholtz_solve_exterior(
                            directions, scattering_obj_i
                        )
                        d_rs_eff[idx_eff, j:j_upper] = u_scat_ext

                    computed_soln_bool = True

                    # This transforms the wave field from (r, s) coordinates to (m, h) coords
                    # as specified by the FY19 paper
                    mh_soln_pre = apply_interp_2d(
                        conv_rs_to_m, conv_rs_to_h, d_rs_eff[idx_eff]
                    ).reshape(num_m, num_h)

                    # Correct for geometric spreading as suggested by FY19
                    d_mh_eff[idx_eff] = mh_soln_pre * get_scale_factor(
                        CONST_RHO_PRIME, CONST_THETA_PRIME
                    )
                except _LinAlgError:
                    d_mh_eff[idx_eff] = np.full_like(d_mh_eff[idx_eff], np.nan)
                    logging.warning("Singular matrix for sample %i", idx_abs)
                    computed_soln_bool = False
                    break

            # Log updates from this sample sample
            sample_completion[idx_abs] = True  # mark as complete :D
            sample_dd = {
                "i": idx_abs,
                "already_present_bool": already_present_bool,
                "computed_soln_bool": computed_soln_bool,
            }
            if not a.dont_use_wandb:
                wandbrun.log(sample_dd)
            # chunk_counter += 1

        # Write results to disk
        logging.warning("Saving data to disk")
        # Write the d_rs and d_mh values to disk then mark sample as complete
        chunk_slice_abs = slice(chunk_start_idx, chunk_end_idx)
        chunk_slice_eff = slice(
            chunk_start_idx - args.start_idx, chunk_end_idx - args.start_idx
        )
        update_field_in_hdf5(
            "q_cart_lpf",
            q_cart_lpf_eff[chunk_slice_eff],
            args.output_fp,
            chunk_slice_abs,
        )
        update_field_in_hdf5(
            "q_polar_lpf",
            q_polar_lpf_eff[chunk_slice_eff],
            args.output_fp,
            chunk_slice_abs,
        )
        update_field_in_hdf5(
            "d_rs", d_rs_eff[chunk_slice_eff], args.output_fp, chunk_slice_abs
        )
        update_field_in_hdf5(
            "d_mh", d_mh_eff[chunk_slice_eff], args.output_fp, chunk_slice_abs
        )
        update_field_in_hdf5(
            "sample_completion",
            sample_completion[chunk_slice_abs],
            args.output_fp,
            chunk_slice_abs,
        )

        try:
            # Also profile GPU ram usage maybe?
            process = psutil.Process()
            logging.warning(f"Memory usage: {process.memory_info().rss >> 20} MB")
            # this is not where the memory usage peaks
            vram_free_bytes, vram_available_bytes = torch.cuda.mem_get_info()
            vram_used_mb = (vram_available_bytes - vram_free_bytes) >> 20
            logging.warning(
                f"Current VRAM usage: {vram_used_mb} MB / {vram_available_bytes>>20} MB"
            )
        except:
            logging.warning(
                f"Skipping memory or VRAM usage calculation due to an error"
            )

    # Fetch from disk again in case someone else was working on the same file at the same time
    sample_completion_newest = load_field_in_hdf5("sample_completion", args.output_fp)
    if np.all(sample_completion_newest):
        # If every sample has been completed then we can mark the file as complete
        # mdf_completion[0] = True
        mdf_completion = np.array([True])
        update_field_in_hdf5(
            "file_completion",
            mdf_completion,
            args.output_fp,
        )
        logging.warning(
            f"Marked the measurement file as complete! ( {args.output_fp} )"
        )
    logging.warning("Finished")
    return


if __name__ == "__main__":
    a = setup_args()

    root = logging.getLogger()

    handler = logging.StreamHandler(sys.stderr)
    if a.debug:
        handler.level = logging.DEBUG
        root.setLevel(logging.DEBUG)
    else:
        handler.level = logging.WARNING
        root.setLevel(logging.WARNING)

    formatter = logging.Formatter(FMT, datefmt=TIMEFMT)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    hash_id = hash_dict(vars(a))

    logging.info(f"Start: generate_measurement_file.py")

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
            try:
                main(a)
            except Exception as e:
                logging.error(f"Fatal Error encountered: {e}")
                logging.error(f"generate_measurements_files.py terminating early")
