"""
This module does not contain runnable tests, but it contains utility functions that other tests call.
It is named 'test_utils.py' so that pytest will import it as a top-level module when running the 
following command from the project root directory.
```
python -m pytest test/
```
"""

import numpy as np
import os
import sys
import pytest
import shutil

FP_TEST_DATA_SWITCHNET = "test/assets/2023-08-11_debugging_dataset_shapes.h5"
FP_TEST_DATA_PARSED = "test/assets/2023-08-11_debugging_dataset_shapes_parsed.h5"

FRMT_TRAIN_MEAS = (
    "test/assets/2024-05-28_tiny_debugging_dataset/train_measurements_nu_{}"
)
FRMT_TEST_MEAS = "test/assets/2024-05-28_tiny_debugging_dataset/test_measurements_nu_{}"
FRMT_VAL_MEAS = "test/assets/2024-05-28_tiny_debugging_dataset/val_measurements_nu_{}"
DIR_TRAIN_SCOBJ = "test/assets/2024-05-28_tiny_debugging_dataset/train_scattering_objs"
DIR_TEST_SCOBJ = "test/assets/2024-05-28_tiny_debugging_dataset/test_scattering_objs"
DIR_VAL_SCOBJ = "test/assets/2024-05-28_tiny_debugging_dataset/val_scattering_objs"

EXPECTED_N_THETA = 192
EXPECTED_N_RHO = 96
EXPECTED_N_X = 192
EXPECTED_N_H = 96
EXPECTED_N_M = 192

WAVENUMBERS_PRESENT = [4, 16]

@pytest.fixture()
def fixture_tmp_dir():
    """This fixture creates a temporary directory for testing purposes. It initializes the directory
    at test/tmp/ and deletes it after the test is run.
    It can be used by importing fixture_tmp_dir into the test file, and then writing a
    test function with argument name fixture_tmp_dir. In the test function,
    the argument fixture_tmp_dir will be the path to the temporary directory.
    """

    d = "test/tmp/"
    if not os.path.isdir(d):
        os.mkdir(d)
    yield d
    shutil.rmtree(d)

def check_arrays_close(
    a: np.ndarray,
    b: np.ndarray,
    a_name: str = "a",
    b_name: str = "b",
    msg: str = "",
    atol: float = 1e-8,
    rtol: float = 1e-05,
) -> None:

    s = _evaluate_arrays_close(a, b, msg, atol, rtol)
    allclose_bool = np.allclose(a, b, atol=atol, rtol=rtol)
    assert allclose_bool, s


def _evaluate_arrays_close(
    a: np.ndarray,
    b: np.ndarray,
    msg: str = "",
    atol: float = 1e-08,
    rtol: float = 1e-05,
) -> str:
    assert a.size == b.size, f"Sizes don't match: {a.size} vs {b.size}"

    max_diff = np.max(np.abs(a - b))
    samp_n = 5

    # Compute relative difference
    x = a.flatten()
    y = b.flatten()
    difference_count = np.logical_not(np.isclose(x, y, atol=atol, rtol=rtol)).sum()

    bool_arr = np.abs(x) >= 1e-15
    rel_diffs = np.abs((x[bool_arr] - y[bool_arr]) / x[bool_arr])
    if bool_arr.astype(int).sum() == 0:
        return msg + "No nonzero entries in A"
    max_rel_diff = np.max(rel_diffs)
    s = (
        msg
        + "Arrays differ in {} / {} entries. Max absolute diff: {}; max relative diff: {}".format(
            difference_count, a.size, max_diff, max_rel_diff
        )
    )

    return s


def check_scalars_close(
    a, b, a_name: str = "a", b_name: str = "b", msg: str = "", atol=1e-08, rtol=1e-05
):
    max_diff = np.max(np.abs(a - b))
    s = msg + "Max diff: {:.8f}, {}: {}, {}: {}".format(max_diff, a_name, a, b_name, b)
    allclose_bool = np.allclose(a, b, atol=atol, rtol=rtol)
    assert allclose_bool, s


def check_no_nan_in_array(arr: np.ndarray) -> None:

    nan_points = np.argwhere(np.isnan(arr))

    s = f"Found NaNs in arr of shape {arr.shape}. Some of the points are at indices {nan_points.flatten()[:5]}"

    assert not np.any(np.isnan(arr)), s
