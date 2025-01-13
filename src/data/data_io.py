"""
This file has helper functions for loading and saving data. See the README in
the root directory for a description about the data saved in hdf5 format, naming
conventions used, and the directory structure of our dataset.

The directory choices were made to facilitate faster IO operations.
"""

from __future__ import annotations
import logging
import h5py
import numpy as np
from typing import Dict, List, Tuple
import os
import time
from src.data.data_naming_constants import (
    KEYS_FOR_TRAINING_SAMPLES_MEAS,
    KEYS_FOR_TRAINING_SAMPLES_ALL,
    KEYS_FOR_TRAINING_METADATA,
    Q_CART,
    Q_POLAR,
    Q_CART_LPF,
    SAMPLE_COMPLETION,
    D_MH,
    D_RS,
    NU_SF,
    OMEGA_SF,
    FREQ_DEPENDENT_KEYS,
    TRUNCATABLE_KEYS,
    KEYS_FOR_EXPERIMENT_INFO_OUT,
)
from src.data.add_noise import add_noise_to_d

import time

# global constant for the number of times the IO functions will re-try accessing a given file
MAX_RETRIES = 10


### Single-file HDF5 access functions ###


def load_hdf5_to_dict(fp_in, key_replacement: dict = None, retries: int = 0) -> Dict:
    """Loads all the fields in a hdf5 file"""
    if retries >= MAX_RETRIES:
        raise IOError(f"(lhtd) Couldn't open load {fp_in} after {MAX_RETRIES} tries")
    key_replacement = key_replacement if key_replacement is not None else {}
    # key replacement function
    krfn = lambda key: key_replacement[key] if key in key_replacement.keys() else key
    # destination dictionary
    data_dict = {}
    try:
        with h5py.File(fp_in, "r") as hf:
            data_dict = {krfn(key): val[()] for (key, val) in hf.items()}
    except BlockingIOError:
        logging.warning(f"File {fp_in} is blocked on attempt {retries}")
        time.sleep(30)
        return load_hdf5_to_dict(fp_in, key_replacement, retries + 1)
    return data_dict


def save_dict_to_hdf5(
    data_dict: Dict,
    fp_out: str,
    key_replacement: dict = None,
    retries: int = 0,
) -> None:
    """Saves a dictionary as a hdf5 file at path fp_out"""
    key_replacement = key_replacement if key_replacement is not None else {}
    # key replacement function
    krfn = lambda key: key_replacement[key] if key in key_replacement.keys() else key

    if retries >= MAX_RETRIES:
        raise IOError(f"(sdth) Couldn't open file {fp_out} after {MAX_RETRIES} tries")
    try:
        with h5py.File(fp_out, "w") as hf:
            for key, val in data_dict.items():
                hf.create_dataset(krfn(key), data=val)
    except BlockingIOError:
        logging.warning(f"File {fp_out} is blocked; {retries} retries remaining")
        time.sleep(30)
        save_dict_to_hdf5(data_dict, fp_out, key_replacement, retries + 1)


### Helper functions to operate on Individual Fields ###
def save_field_to_hdf5(
    key: str, data: np.ndarray, fp_out: str, overwrite: bool = True, retries: int = 0
) -> None:
    """Saves an individual array to the specified field in a given hdf5 file
    Note that this operation may squash the old field
    """
    if not os.path.exists(fp_out):
        raise FileNotFoundError
    if retries >= MAX_RETRIES:
        raise IOError(f"(sfth) Couldn't open file {fp_out} after {MAX_RETRIES} tries")
    try:
        with h5py.File(fp_out, "a") as hf:
            # logging.debug(f"sfth df keys before: {hf.keys()}")
            if key in hf.keys() and not overwrite:
                raise KeyError(
                    f"Attempted to write to key {key} in {fp_out} "
                    "which already exists (and overwrite mode=False)"
                )
            elif key in hf.keys() and overwrite:
                # Need to handle the case where the dataset already exists...
                # See the update_field_in_hdf5 function for ideas maybe?
                # pass
                dset = hf.require_dataset(key, shape=data.shape, dtype=data.dtype)
                dset.write_direct(data)
            else:
                # Create new entry
                hf.create_dataset(key, data=data)
            # logging.debug(f"sfth df keys after:  {hf.keys()}")

    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(30)
        save_field_to_hdf5(key, data, fp_out, retries + 1)


# Provide an alias for naming consistency but leave the original version
# intact to avoid breaking anything
# save_field_in_hdf5 = save_field_to_hdf5


def save_field_in_hdf5(
    key: str, data: np.ndarray, fp_out: str, overwrite: bool = True, retries: int = 0
) -> None:
    """Saves an individual array to the specified field in a given hdf5 file
    Note that this operation may squash the old field!

    Alias for save_field_to_hdf5 for better consistency
    with the other field-specific helper functions
    """
    save_field_to_hdf5(key, data, fp_out, overwrite=overwrite, retries=retries)


def load_field_in_hdf5(
    key: str, fp_out: str, idx_slice=slice(None), retries: int = 0
) -> np.ndarray:
    """Loads an individual field to the specified field in a given hdf5 file"""
    if not os.path.exists(fp_out):
        raise FileNotFoundError("Can't load field %s from %s" % (key, fp_out))
    if retries >= MAX_RETRIES:
        raise IOError(f"(lfih) Couldn't open file after {MAX_RETRIES} tries")
    try:
        with h5py.File(fp_out, "r") as hf:
            data_loaded = hf[key][()]
            data = data_loaded[idx_slice]

        return data

    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(30)
        return load_field_in_hdf5(key, fp_out, idx_slice, retries + 1)


def update_field_in_hdf5(
    key: str, data: np.ndarray, fp_out: str, idx_slice=slice(None), retries: int = 0
) -> None:
    """Saves an individual array to a slice in the specified field in a given hdf5 file
    Note that this operation may squash the old field
    """
    if not os.path.exists(fp_out):
        raise FileNotFoundError
    if retries >= MAX_RETRIES:
        raise IOError(f"(ufih) Couldn't open file after {MAX_RETRIES} tries")

    try:
        with h5py.File(fp_out, "a") as hf:
            data_loaded = hf[key][()]
            data_loaded[idx_slice] = data
            dset = hf.require_dataset(
                key, shape=data_loaded.shape, dtype=data_loaded.dtype
            )
            dset.write_direct(data_loaded)
    except KeyError:
        # In case the field was not located, just make a new one...
        save_field_to_hdf5(key, data, fp_out, retries)

    except BlockingIOError:
        logging.warning("File is blocked; on retry # %i", retries)
        time.sleep(30)
        update_field_in_hdf5(key, data, fp_out, idx_slice, retries + 1)


### Directory-wide HDF5 loading function ###


# Define a custom sorting key function
def _get_number_from_filename(filename: str) -> int:
    """Assumes a file has format .*_{number}.h5 and extracts the number"""
    f = filename.split("_")[-1]
    return int(f.split(".")[0])


def load_dir(
    meas_dir_name: str,
    scobj_dir_name: str,
    truncate_num: int | None = None,
    load_cart_and_rs: bool = False,
) -> Dict[str, np.ndarray]:
    """Loads the data from a directory of hdf5 files that we assume to have the same fields, as specified by the generate_measurement_files.py script.

    Args:
        meas_dir_name (str): Directory containing all of the measurement files
        scobj_dir_name (str): Directory containing all of the scattering object files
        truncate_num (int | None, optional): How many samples to load. If set to None, all samples are loaded. Defaults to None.
        load_cart_and_rs (bool, optional): whether to load q_cart, q_cart_lpf and d_rs

    Returns:
        Dict[str, np.ndarray]: contains all keys in the union of KEYS_FOR_TRAINING_SAMPLES and KEYS_FOR_TRAINING_METADATA
    """
    # 1. Determine the relevant files to be loaded

    # Get the list of measurement files
    file_list = os.listdir(meas_dir_name)
    file_list = sorted(file_list, key=_get_number_from_filename)

    # Get the list of scattering object files
    file_list_scobj = os.listdir(scobj_dir_name)
    file_list_scobj = sorted(file_list_scobj, key=_get_number_from_filename)

    assert len(file_list) == len(
        file_list_scobj
    ), "Can't load directories with different lengths: %i vs %i, %s vs %s" % (
        len(file_list),
        len(file_list_scobj),
        meas_dir_name,
        scobj_dir_name,
    )
    logging.info(f"(load_dir) preparing to load {len(file_list)} files...")

    n_files = len(file_list)

    # 2. Select which fields should be loaded
    # Keys to be appended; also a subset for extracting from the measurement files
    keys_to_append = [*KEYS_FOR_TRAINING_SAMPLES_ALL]  # copy to avoid over-writing it
    keys_to_append_from_meas = [*KEYS_FOR_TRAINING_SAMPLES_MEAS]

    if load_cart_and_rs:
        keys_to_append += [D_RS, Q_CART, Q_CART_LPF]  # include q_cart_lpf
        keys_to_append_from_meas += [D_RS, Q_CART_LPF]
        keys_to_ignore = []
    else:
        keys_to_ignore = [D_RS, Q_CART, Q_CART_LPF]

    # 3. Load the first file to determine the appropriate shapes for the fields
    fp_0_meas = os.path.join(meas_dir_name, file_list[0])
    out_dd = load_hdf5_to_dict(fp_0_meas)
    if not load_cart_and_rs:
        for kti in keys_to_ignore:
            if kti in out_dd.keys():
                del out_dd[kti]
    # n_samples_0 = out_dd[D_MH].shape[0]
    n_samples_0 = out_dd[Q_POLAR].shape[0]

    fp_0_scobj = os.path.join(scobj_dir_name, file_list_scobj[0])
    out_dd[Q_POLAR] = load_field_in_hdf5(Q_POLAR, fp_0_scobj)
    logging.info(f"(load_dir) first file has been prepared")

    # Set truncate_num to infinity if not specified.
    truncate_num = np.inf if truncate_num is None else truncate_num

    # If we already have enough samples, the loading process is already finished
    # So, exit early
    if n_samples_0 > truncate_num:
        for k in keys_to_append:
            if k in out_dd:
                out_dd[k] = out_dd[k][:truncate_num]
        return out_dd

    # 4. Append the relevant fields for the rest of the files in the directory
    for i in range(1, n_files):
        break_bool = False
        fname = file_list[i]
        fname_scobj = file_list_scobj[i]
        fp_meas = os.path.join(meas_dir_name, fname)
        fp_scobj = os.path.join(scobj_dir_name, fname_scobj)
        # Temporary dictionary that just holds the fields to be extended
        dd_new = {
            key: load_field_in_hdf5(key, fp_meas) for key in keys_to_append_from_meas
        }
        dd_new[Q_POLAR] = load_field_in_hdf5(Q_POLAR, fp_scobj)
        if load_cart_and_rs:
            dd_new[Q_CART] = load_field_in_hdf5(Q_CART, fp_scobj)

        new_n_samples = dd_new[SAMPLE_COMPLETION].shape[0]  # number of new samples

        # print(f"dd_new keys: {dd_new.keys()}")
        # print(f"keys_to_append: {keys_to_append}")
        # Check whether to truncate here
        if out_dd[SAMPLE_COMPLETION].shape[0] + new_n_samples > truncate_num:
            # In the case that we have to truncate, we first compute
            # how many samples to keep, and then concatenate the contents
            # of dd_new into out_dd
            n_samples_to_keep = truncate_num - out_dd[SAMPLE_COMPLETION].shape[0]
            dd_new = {key: dd_new[key][:n_samples_to_keep] for key in keys_to_append}
            break_bool = True

        # logging.debug("load_dir: Here are all of the keys present: %s", dd_new.keys())
        # logging.debug(
        #     "load_dir: Here is the shape of q_polar: %s", dd_new[Q_POLAR].shape
        # )

        for key in keys_to_append:
            out_dd[key] = np.concatenate(
                [out_dd[key], dd_new[key]]
            )  # concatenates along axis 0
        if break_bool:
            # Break out of the for loop if we have to truncate
            break
    # logging.debug("load_dir: Here are all of the keys present: %s", out_dd.keys())
    # logging.debug("load_dir: Here is the shape of q_polar: %s", out_dd[Q_POLAR].shape)

    # If we choose not to load q_cart or d_rs, then we delete the fields entirely to avoid confusion
    if not load_cart_and_rs:
        if Q_CART_LPF in out_dd.keys():
            del out_dd[Q_CART_LPF]
        if Q_CART in out_dd.keys():
            del out_dd[Q_CART]
        if D_RS in out_dd.keys():
            del out_dd[D_RS]

    return out_dd


# Helper function to process the loaded directory
def load_multifreq_dataset(
    freq_dir_list: List[str],
    truncate_num: int = None,
    key_replacement: dict = None,
    noise_to_sig_ratio: float = None,
    add_noise_to: str = None,
    load_cart: bool = False,
    nan_mode: str = None,
) -> Tuple[Dict, Dict]:
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
            load_cart_and_rs=load_cart,
        )
        dd_list.append(dd_new)
    logging.info(f"Finished loading the dataset")

    # Set up the dictionary for fixed values and fields that will get multiple frequencies
    dd_all = dict()
    simple_fdk_list = []
    present_fdk_list = []
    for key, val in dd_list[0].items():
        if key not in FREQ_DEPENDENT_KEYS:  # or key in [OMEGA_SF, NU_SF]:
            # If the key is not frequency-dependent we only need this one value
            dd_all[key] = val
        elif key in {OMEGA_SF, NU_SF}:
            # For omega_sf and nu_sf, we just need a scalar per frequency
            dd_all[key] = np.zeros(N_freqs)
            simple_fdk_list.append(key)
        else:
            # All other keys with frequency dependence will need extra space
            # e.g., d_mh needs an index per frequency
            # Assume we always just put the new frequency index in dim 1
            curr_shape = dd_list[0][key].shape
            logging.info(f"Found fdk {key} whose entry has shape {curr_shape}")
            new_shape = tuple([*curr_shape[:1], N_freqs, *curr_shape[1:]])
            dd_all[key] = np.zeros(new_shape, dtype=dd_list[0][key].dtype)
            present_fdk_list.append(key)

    # For each frequency-dependent key in the loaded file, fetch the data from other frequencies
    for fdk in present_fdk_list:
        # Fetch the relevant slice for each frequency (and for each frequency-dependent key)
        for fi in range(N_freqs):
            logging.info(
                f"(key {fdk}) Loading value of shape {dd_list[fi][fdk].shape} into a slice"
                f" of {dd_all[fdk].shape}"
            )
            dd_all[fdk][:, fi] = dd_list[fi][fdk]
    for simple_fdk in simple_fdk_list:
        # Repeat for scalars (i.e., nu_sf and omega_sf)
        for fi in range(N_freqs):
            dd_all[simple_fdk][fi] = dd_list[fi][simple_fdk].item()

    # Decide how to deal with with NaNs; determine using d_mh
    nan_mode = nan_mode.lower() if nan_mode is not None else "skip"
    num_nan_samples = np.sum(np.any(np.isnan(dd_all[D_MH]), axis=(-3, -2, -1)))
    aux_dd = {
        "num_nan_samples": num_nan_samples,
        "num_loaded_samples": dd_all[D_MH].shape[0],
        "num_valid_samples": dd_all[D_MH].shape[0] - num_nan_samples,
        "orig_idcs": np.full(dd_all[D_MH].shape[0], True, dtype=bool),
        # "orig_idcs": np.arange(dd_all[D_MH].shape[0]),
    }
    our_keys_for_training_samples_all = [
        *KEYS_FOR_TRAINING_SAMPLES_ALL,
        Q_CART,
        Q_CART_LPF,
    ]

    def get_valid_idcs(arr):
        return np.logical_not(np.any(np.isnan(arr), axis=tuple(range(1, arr.ndim))))

    # Make sure everything is processed together re nans
    keep_idcs = np.logical_and.reduce(
        [get_valid_idcs(dd_all[key]) for key in TRUNCATABLE_KEYS if key in dd_all]
    )
    nan_idcs = np.logical_not(keep_idcs)
    aux_dd["orig_idcs"] = keep_idcs
    if nan_mode == "keep":
        # Keep the NaNs
        # for key in TRUNCATABLE_KEYS:
        #     if key in dd_all.keys() and key in our_keys_for_training_samples_all:
        #         dd_all[key] = np.nan_to_num(dd_all[key], copy=False, nan=np.nan)
        pass
    elif nan_mode == "zero":
        # Zero out for everything
        for key in TRUNCATABLE_KEYS:
            if key in dd_all.keys() and key in our_keys_for_training_samples_all:
                # dd_all[key] = np.nan_to_num(dd_all[key], copy=False, nan=0)
                dd_all[key][nan_idcs] = 0
    elif nan_mode == "skip":
        # wave_field_mh shape: (N_samples, N_freqs, N_m, N_h)
        for key in TRUNCATABLE_KEYS:
            if key in dd_all.keys() and key in our_keys_for_training_samples_all:
                dd_all[key] = dd_all[key][keep_idcs]
        # Original indices of the included samples
        logging.info(f"orig_idcs: {len(keep_idcs)} entries")
    logging.info(f"num_loaded_samples: {aux_dd['num_loaded_samples']}")
    logging.info(f"num_valid_samples: {aux_dd['num_valid_samples']}")
    logging.info(f"num_nan_samples: {aux_dd['num_nan_samples']}")
    logging.info(f"dd_all[d_mh]: {dd_all[D_MH].shape}")

    # Add noise if applicable
    if noise_to_sig_ratio is not None:
        add_noise_to = add_noise_to.lower() if add_noise_to is not None else "d_mh"
        if add_noise_to == "d_mh":
            dd_all[D_MH] = add_noise_to_d(dd_all[D_MH], noise_to_sig_ratio)
        elif add_noise_to == "d_rs":
            dd_all[D_RS] = add_noise_to_d(dd_all[D_RS], noise_to_sig_ratio)
        elif add_noise_to == "none":
            pass
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

    # The metadata should be contained in dd_all
    # But hopefully this helps compatibility with other
    metadata_dd = {
        key: dd_all[key] for key in KEYS_FOR_EXPERIMENT_INFO_OUT if key in dd_all.keys()
    }
    dd_all = {**aux_dd, **dd_all}
    metadata_dd = {**aux_dd, **metadata_dd}

    return dd_all, metadata_dd
