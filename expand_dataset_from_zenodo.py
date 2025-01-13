import os
import sys
import h5py
import argparse
import logging
from src.utils.logging_utils import FMT, TIMEFMT
from src.data.data_naming_constants import (
    Q_CART,
    Q_POLAR,
    X_VALS,
    RHO_VALS,
    THETA_VALS,
    SEED,
    CONTRAST,
    BACKGROUND_MAX_FREQ,
    NUM_SHAPES,
    GAUSSIAN_LPF_PARAM,
    SAMPLE_COMPLETION,
    FILE_COMPLETION,
)

# List of keys to include in the new HDF5 files
K_INCLUDE_SCAT_OBJ_FILE = [
    Q_CART,
    Q_POLAR,
    X_VALS,
    RHO_VALS,
    THETA_VALS,
    SEED,
    CONTRAST,
    BACKGROUND_MAX_FREQ,
    NUM_SHAPES,
    GAUSSIAN_LPF_PARAM,
    SAMPLE_COMPLETION,
    FILE_COMPLETION,
]


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand the dataset from Zenodo")
    parser.add_argument(
        "-data_dir",
        type=str,
        help="Path to the directory containing the dataset",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:

    for dset_type in ["train", "val", "test"]:

        measurements_dir = os.path.join(args.data_dir, f"{dset_type}_measurements_nu_1")

        # Check to make sure the measurements directory exists
        if not os.path.exists(measurements_dir):
            raise ValueError(f"Measurements directory not found: {measurements_dir}")

        scattering_objs_dir = os.path.join(
            args.data_dir, f"{dset_type}_scattering_objs"
        )

        # Create the output directory if it doesn't exist
        os.makedirs(scattering_objs_dir, exist_ok=True)

        # List all files in the measurements directory
        files = [f for f in os.listdir(measurements_dir) if f.endswith(".h5")]

        for file_name in files:
            input_path = os.path.join(measurements_dir, file_name)

            # Get the sample index from the file name
            file_idx = file_name.split("_")[-1].rstrip(".h5")
            output_path = os.path.join(
                scattering_objs_dir, f"scattering_objs_{file_idx}.h5"
            )
            logging.info("Writing to %s", output_path)

            with h5py.File(input_path, "r") as input_file:
                with h5py.File(output_path, "w") as output_file:
                    for key in K_INCLUDE_SCAT_OBJ_FILE:
                        if key in input_file:
                            input_file.copy(key, output_file)
                        else:
                            raise ValueError(f"Key {key} not found in {input_path}")
    logging.info("Finished")


if __name__ == "__main__":
    logging.basicConfig(format=FMT, datefmt=TIMEFMT, level=logging.INFO)
    a = setup_args()
    main(a)
