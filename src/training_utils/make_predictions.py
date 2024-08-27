from typing import Dict, Callable
import torch
import numpy as np
import os
import logging
from src.data.data_io import save_dict_to_hdf5
from src.data.data_naming_constants import (
    Q_CART,
    Q_POLAR,
    THETA_VALS,
    RHO_VALS,
    X_VALS,
    SAMPLE_COMPLETION,
)
from src.data.data_transformations import (
    prep_polar_padder,
    polar_pad_and_apply,
    prep_conv_interp_2d,
)

FMT_STR = "scattering_objs_{}.h5"


def make_preds_on_dataset(
    model: torch.nn.Module,
    dloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    shard_size: int,
    experiment_info: Dict[str, np.ndarray],
    format_str: str = FMT_STR,
) -> None:
    """
    This function makes predictions on the dataset in dloader, converts them
    to cartesian coordinates, and then saves the dataset of predictions to
    different files, using the output_dir, format_str and shard_size.
    The predictions are saved using save_dict_to_hdf5.

    Args:
        model (torch.nn.Module): Model with trained parameters
        dloader (torch.utils.data.DataLoader): Data loader for the dataset
        device (torch.device): Device where the model is located
        output_dir (str): Where to save the data
        shard_size (int): Number of predictions to save in each file
        conv_filter_to_x (np.ndarray): The convolution filter to convert from
            polar to cartesian coordinates in the x direction
        conv_filter_to_y (np.ndarray): The convolution filter to convert from
            polar to cartesian coordinates in the y direction
        experiment_info (Dict[str, np.ndarray]): A dictionary containing the
            metadata such as x_vals, rho_vals, theta_vals, etc.
        format_str (str, optional): The format string for the output files.

    Returns:
        None
    """
    preds_polar = torch.zeros(
        (
            len(dloader.dataset),
            experiment_info[THETA_VALS].shape[0],
            experiment_info[RHO_VALS].shape[0],
        ),
        dtype=torch.float32,
    )
    model.eval()
    with torch.no_grad():
        for i, (x, y, z) in enumerate(dloader):
            x = x.to(device)

            n_samples = x.shape[0]

            preds = model(x)
            # This happens in the case when we're making predictions with a ResidualFYNetInverse model, which
            # makes a prediction for each input wave frequency
            if preds.dim() == 4:
                preds = preds[:, -1]
            nn = i * dloader.batch_size
            preds_polar[nn : nn + n_samples] = preds.to("cpu")

    # Now, convert the predictions to cartesian coordinates.
    to_cart_fn = prepare_polar_to_cart(
        experiment_info[X_VALS], experiment_info[THETA_VALS], experiment_info[RHO_VALS]
    )
    preds_cart = torch.zeros(
        (
            len(dloader.dataset),
            experiment_info[X_VALS].shape[0],
            experiment_info[X_VALS].shape[0],
        ),
        dtype=torch.float32,
    )
    for i in range(0, len(preds_polar), shard_size):
        preds_cart[i : i + shard_size] = to_cart_fn(preds_polar[i : i + shard_size])

    if output_dir is None:
        return preds_cart.numpy(), preds_polar.numpy()

    # Finally, save the predictions to different files.
    sample_completion = experiment_info[SAMPLE_COMPLETION]
    preds_polar = preds_polar.numpy()
    preds_cart = preds_cart.numpy()
    for i in range(0, len(dloader.dataset), shard_size):
        shard_cart = preds_cart[i : i + shard_size]
        shart_polar = preds_polar[i : i + shard_size]
        sample_completion_shard = sample_completion[i : i + shard_size]

        out_dd = {
            Q_CART: shard_cart,
            Q_POLAR: shart_polar,
            SAMPLE_COMPLETION: sample_completion_shard,
        }

        experiment_info.update(out_dd)
        # out_dd.update(experiment_info)
        out_fp = os.path.join(output_dir, format_str.format(i))
        save_dict_to_hdf5(
            experiment_info,
            out_fp,
        )


def prepare_polar_to_cart(
    x_vals: np.ndarray, theta_vals: np.ndarray, rho_vals: np.ndarray
) -> Callable:
    """Prepare the polar-to-cartesian interpolation operators"""
    center = np.zeros(2)  # could shift around the center later if we wanted

    N_rho = rho_vals.shape[0]
    N_theta = theta_vals.shape[0]
    N_x = x_vals.shape[0]
    N_y = x_vals.shape[0]

    data_grid_xy = (
        np.array(np.meshgrid(x_vals, x_vals)).transpose(1, 2, 0).reshape(N_x * N_y, 2)
    )
    cart_grid_radii = np.sqrt(data_grid_xy[:, 0] ** 2 + data_grid_xy[:, 1] ** 2)
    cart_grid_thetas = np.mod(
        np.arctan2(data_grid_xy[:, 1] - center[1], data_grid_xy[:, 0] - center[0]),
        2 * np.pi,
    )
    cart_grid_polar_coords = np.array([cart_grid_thetas, cart_grid_radii]).T

    padded_rho_vals, polar_padder = prep_polar_padder(
        rho_vals, N_theta, dim=1, with_torch=True
    )

    polar_to_x, polar_to_y = prep_conv_interp_2d(
        theta_vals,
        padded_rho_vals,
        cart_grid_polar_coords,
        bc_modes=("periodic", "extend"),
        # bc_modes=("extend", "periodic"),
        a_neg_half=True,
    )
    # Convert the operations to pytorch formats
    # polar_to_x = torch.sparse_csr_tensor(
    #     polar_to_x.indptr,
    #     polar_to_x.indices,
    #     polar_to_x.data,
    #     dtype=torch.float32,
    #     requires_grad=False,
    # )
    polar_to_x = torch.tensor(
        polar_to_x.todense(), dtype=torch.float32, requires_grad=False
    )
    polar_to_y = torch.tensor(
        polar_to_y.todense(), dtype=torch.float32, requires_grad=False
    )

    def my_polar_to_cart(polar_data: np.ndarray) -> np.ndarray:
        """Helper function to send polar grid data to cartesian grid data
        Explicitly grabs values outside this function to reduce the chances of a weird scoping issue
        Note that this helper function also takes care of reshaping
        """
        res = polar_pad_and_apply(
            polar_padder,
            polar_to_x,
            polar_to_y,
            polar_data.reshape(-1, N_theta, N_rho),
            batched=True,
        ).reshape(
            -1, N_y, N_x
        )  # .transpose(1, 2)

        return res

    return my_polar_to_cart
