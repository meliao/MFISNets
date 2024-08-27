import hashlib
import os
import logging
import numpy as np
from typing import Any, Dict, Tuple


FMT = "%(asctime)s:MFISNets: %(levelname)s - %(message)s"
TIMEFMT = "%Y-%m-%d %H:%M:%S"


def hash_dict(dictionary: Dict[str, Any]) -> str:
    """Create a hash for a dictionary."""
    dict2hash = ""

    for k in sorted(dictionary.keys()):
        if isinstance(dictionary[k], dict):
            v = hash_dict(dictionary[k])
        else:
            v = dictionary[k]

        dict2hash += "%s_%s_" % (str(k), str(v))

    return hashlib.md5(dict2hash.encode()).hexdigest()


def write_result_to_file(fp: str, missing_str: str = "", **trial) -> None:
    """Write a line to a tab-separated file saving the results of a single
        trial.
    Parameters
    ----------
    fp : str
        Output filepath
    missing_str : str
        (Optional) What to print in the case of a missing trial value
    **trial : dict
        One trial result. Keys will become the file header
    Returns
    -------
    None
    """
    header_lst = list(trial.keys())
    header_lst.sort()
    if not os.path.isfile(fp):
        header_line = "\t".join(header_lst) + "\n"
        with open(fp, "w") as f:
            f.write(header_line)
    trial_lst = [str(trial.get(i, missing_str)) for i in header_lst]
    trial_line = "\t".join(trial_lst) + "\n"
    with open(fp, "a") as f:
        f.write(trial_line)


def extract_line_by_field(
    file_name: str,
    field: str,
    selection_mode: str = "min",
) -> Tuple[Dict, float]:
    """
    Takes a tab-separated file and extracts the line containing the min/max value of a given field.

    This is used to find the best epoch in a training log, for example.
    Parameters:
        file_name (string/file path): name of the relevant file to retrieve
        field (string): name of the field in question
        selection_mode (string): whether to choose the line with minimum/maximum field value
        verbosity_level (int): indicate a relative level of outputs
    Return Value:
        line_entry (Dict): a lookup-table of the contents in this particular line (to avoid
            concerns about ordering within the header)
        field_value_selected (int/float most likely): the relevant min/max value of the field in question
    """
    with open(file_name, "r") as file:
        file_contents = [line.strip().split("\t") for line in file]
    header = file_contents[0]
    contents = file_contents[1:]

    try:
        field_idx = header.index(field)
    except:
        raise KeyError(f"Unable to locate field '{field}' in the header {header}")
    field_arr = np.array([parse_val(entry[field_idx]) for entry in contents])

    if selection_mode.lower() == "min":
        line_idx = np.argmin(field_arr)
    elif selection_mode.lower() == "max":
        line_idx = np.argmax(field_arr)
    else:
        raise ValueError(
            f"Expected mode keyword as one of ['min', 'max'] to choose the selection direction"
        )
    field_val_selected = field_arr[line_idx]
    line_entry = {
        key: parse_val(contents[line_idx][ki]) for ki, key in enumerate(header)
    }

    return line_entry, field_val_selected


def parse_val(text_val):
    """Parses a text to int, float, or bool if possible"""
    try:
        return int(text_val)
    except:
        pass
    try:
        return float(text_val)
    except:
        pass
    if text_val in ["True", "true"]:
        return True
    elif text_val in ["False", "false"]:
        return False


def find_best_epoch(
    results_fp: str, val_error_field: str, selection_mode: str = "min"
) -> Dict:
    """
    Find the epoch with the best validation error in a training log.
    Parameters:
        results_fp (str): path to the training log
        val_error_field (str): name of the validation error field in the log
        selection_mode (str): whether to choose the line with minimum/maximum validation error
    Return Value:
        (Dict): the key-value mapping of the best epoch's contents
    """
    line_entry, val_error = extract_line_by_field(
        results_fp, val_error_field, selection_mode=selection_mode
    )
    return line_entry
