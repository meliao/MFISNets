from typing import Callable, Dict
from timeit import default_timer
import logging
import torch
import numpy as np


def train(
    model: torch.nn.Module,
    n_epochs: int,
    lr_init: float,
    weight_decay: float,
    momentum: float,
    eta_min: float,
    train_loader: torch.utils.data.DataLoader,
    device: torch.cuda.Device,
    n_epochs_per_log: int,
    log_function: Callable = None,
    loss_function: Callable = None,
    use_cart_output: bool = False,
    # reweighted_polar: bool = False,
) -> torch.nn.Module:
    """A general-purpose training script using Adam as the optimizer and
    CosineAnnealingLR as the LR scheduler

    Args:
        model (torch.nn.Module): Model with trainable parameters
        n_epochs (int): Number of epochs to do training. No early stopping.
        lr_init (float): Initial learning rate
        weight_decay (float): Amount of L2 regularization applied
        momentum (float): Amount of momentum applied (NOT USED)
        eta_min (float): A parameter for the CoseineLR
        train_loader (torch.utils.data.DataLoader): Pytorch data loader for the training set
        device (torch.cuda.Device): Specify which device to perform the training on
        n_epochs_per_log (int): How often to run the logging function
        log_function (Callable, optional): Pass an optional function that is
            called every n_epochs_per_logs epochs
            e.g. can log the training progress, save model weights, and output to a file.
            Defaults to None.
        use_cart_output (bool): whether to use the cartesian output of the neural network for the loss function

    Returns:
        torch.nn.Module: _description_
    """
    if n_epochs == 0:
        return model.to("cpu")

    model = model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr_init,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=eta_min
    )
    # logging.info("Beginning model training for %i epochs", n_epochs)

    # loss_fn_name = (
    #     "cartesian"
    #     if use_cart_output
    #     else ("polar" if not reweighted_polar else "reweighted-polar")
    # )
    loss_fn_name = "cartesian"
    logging.info("Using dataset: %s", train_loader.dataset)
    logging.info("Training using loss function/module: %s", loss_function)
    logging.info("Training model: %s", model)
    logging.info("Using LR: %f and min LR: %f", lr_init, eta_min)
    t1 = default_timer()

    model = model.to(device)
    for epoch in range(n_epochs):

        running_sum_squared_error_polar = 0
        running_sum_squared_error_cart = 0

        for i, data_i in enumerate(train_loader):
            x = data_i[0]
            # logging.debug("train: x shape: %s", x.shape)
            y = data_i[1]
            y_final = data_i[2]
            # logging.debug("train: y shape: %s", y.shape)

            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            model_output = model(x)
            if isinstance(model_output, tuple):
                # Choose between polar/cartesian
                pred_polar, pred_cart = model_output
                pred_for_loss = pred_cart if use_cart_output else pred_polar
            else:
                pred_polar = model_output
                pred_for_loss = model_output

            loss_val = loss_function(pred_for_loss, y)
            loss_val.backward()
            optimizer.step()

            if log_function is None:
                with torch.no_grad():
                    running_sum_squared_error_polar += x.shape[0] * loss_function(
                        pred_polar, y
                    )

        scheduler.step()

        if epoch % n_epochs_per_log == 0:
            if log_function is not None:
                log_function(model, epoch)
            else:
                epoch_mse_polar = running_sum_squared_error_polar / len(
                    train_loader.dataset
                )

                logging.info(
                    f"Epoch {epoch} / {n_epochs}. Polar mse: {epoch_mse_polar:.3e}"
                )
    if log_function is not None:
        log_function(model, epoch)
    t2 = default_timer()
    logging.info("Optimization is complete in %f seconds", t2 - t1)
    return model.to("cpu")


def evaluate_losses_on_dataloader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn_dict: Dict,
    device: torch.cuda.Device,
) -> Dict:
    """Evaluate a collection of loss functions for a given model and dataset
    Args:
        model (torch.nn.Module): pytorch model in question
        loader (torch.utils.data.DataLoader): data loader, e.g. for the training or validation set
        loss_fn_dict (Dict of loss functions): collection of loss functions to evaluate
        device (torch.cuda.Device): device on which to perform the evaluations
    Returns:
        out_dd (Dict): dictionary of the loss function values for each item in the dataset
            This function does not perform the aggregation down to a scalar value
            for greater flexibility (e.g., can find the mean, std, median outside
            this function as needed)
    """
    model = model.to(device)
    model.eval()
    n_samples = len(loader.dataset)
    n_batch = loader.batch_size

    out_dd = {
        k: torch.zeros(n_samples, dtype=torch.float32) for k in loss_fn_dict.keys()
    }

    for i, (x, y, y_final) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        y_final = y_final.to(device)

        n_samples = x.shape[0]

        preds = model(x)

        for loss_key, loss_fn in loss_fn_dict.items():
            out_dd[loss_key][i * n_batch : (i * n_batch) + n_samples] = loss_fn(
                preds, y, y_final
            ).cpu()

    return out_dd


def evaluate_losses_on_dataloader_cart_polar(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn_dict: Dict,
    device: torch.cuda.Device,
    extra_output: bool = False,
) -> Dict:
    """Helper function to evaluate loss functions with batched execution (using the data loader)
    Modified from the original from Owen to handle the mixed cartesian/polar targets

    Parameters:
        model (torch.nn.Module): pytorch (FYNet) model that returns polar and cartesian outputs
        loader (torch.utils.data.DataLoader): pytorch data loader for easy batched data access
        loss_fn_dict (Dict of functions of type (tensor, tensor, tensor, tensor) -> array):
            dictionary of functions to calculate the error on a batch of examples compared to the reference
            takes in (pred_polar, pred_cart, yp [y_polar], yc [y_cart]) and returns an array with floats for each batched example
        device (torch.cuda.Device): cpu/gpu device to perform the evaluation on
        extra_output (bool): whether to output some of the intermediate results to the log
    Returns:
        out_dd (dictionary of scalars)
    """

    model = model.to(device)
    model = model.eval()
    tot_samples = len(loader.dataset)
    batch_size = loader.batch_size

    intermediate_dd = {
        k: np.zeros(tot_samples, dtype=np.float32) for k in loss_fn_dict.keys()
    }

    with torch.no_grad():
        for i, (x, yp, yc) in enumerate(loader):
            n_samples = x.shape[0]
            x = x.to(device)
            yp = yp.to(device)
            yc = yc.to(device)

            preds_polar, preds_cart = model(x)

            for loss_key, loss_fn in loss_fn_dict.items():
                intermediate_dd[loss_key][
                    i * batch_size : i * batch_size + n_samples
                ] = loss_fn(preds_polar, preds_cart, yp, yc).cpu()

    out_dd = {key: val.mean() for (key, val) in intermediate_dd.items()}
    return out_dd
