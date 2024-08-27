import torch
import logging


def mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    loss_mse = torch.mean(torch.square(preds.flatten() - targets.flatten()))
    return loss_mse


def _mse_along_batch(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    squared_errors = torch.square(preds - targets)
    means_along_batch = torch.mean(torch.flatten(squared_errors, start_dim=1), dim=1)
    return means_along_batch


def _range_along_batch(preds: torch.Tensor) -> torch.Tensor:
    preds_2d = torch.flatten(preds, start_dim=1)
    maxes = torch.max(preds_2d, dim=1)[0]
    mins = torch.min(preds_2d, dim=1)[0]
    return maxes - mins


def psnr_complex(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Takes in 2 complex arrays, casts them as real, and then computes the
    PSNR across the batch dimension

    PSNR = 10 * log10(targets_range^2 / MSE)
    Args:
        preds (torch.Tensor): Has shape (n_batch, *) and a complex datatype
        targets (torch.Tensor): Has shape (n_batch, *) and a complex datatype

    Returns:
        torch.Tensor: Real scalar
    """
    preds_real = torch.view_as_real(preds)
    targets_real = torch.view_as_real(targets)
    return psnr(preds_real, targets_real)


def psnr(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes the PSNR across the batch dimension

    PSNR = 10 * log10(targets_range^2 / MSE)
    Args:
        preds (torch.Tensor): Has shape (n_batch, *) and a complex datatype
        targets (torch.Tensor): Has shape (n_batch, *) and a complex datatype

    Returns:
        torch.Tensor: Has shape (n_batch,)
    """
    mses_log = torch.log10(_mse_along_batch(preds, targets))
    ranges_log = torch.log10(_range_along_batch(targets))
    ranges_log_sq = 2 * ranges_log

    out = 10 * (ranges_log_sq - mses_log)
    return out


def relative_l2_error(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(preds):
        preds = torch.view_as_real(preds)
        targets = torch.view_as_real(targets)

    diff_l2_norms = torch.norm(torch.flatten(targets - preds, start_dim=1), dim=1)
    target_l2_norms = torch.norm(torch.flatten(targets, start_dim=1), dim=1)

    diff_log = torch.log(diff_l2_norms)
    target_log = torch.log(target_l2_norms)

    return torch.exp(diff_log - target_log)


class MSEModule(torch.nn.Module):
    """I use this in the multi-frequency network training script when I want to
    be able to parameterize the loss function to only pay attention to certain parts
    of the target / predictions.

    The multi-frequency network outputs arrays with shape
    (N_batch, N_freqs, N_theta, N_rho), and the datasets
    have "ground truth" targets with the same shape.

    For example, in training phase 1, we want to compare
    preds[:, 0] against targets[:, 0]. This is accomplished by initializing this class
    with loss_idx = 0.
    """

    def __init__(
        self,
    ) -> None:
        """Initializes the class.

        Args:
            loss_idx (int): The index of the frequency axis (2nd axis) which is used in the
            loss functions.
            final_output_idx (int): The final index of the frequency axis. This is used to
            compute *_against_final losses.
        """
        super().__init__()

    def psnr(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return psnr(preds, targets)

    def relative_l2_error(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return relative_l2_error(preds, targets)

    def mse(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return _mse_along_batch(preds, targets)

    def psnr_against_final(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return psnr(preds, targets_final)

    def relative_l2_error_against_final(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return relative_l2_error(preds, targets_final)

    def mse_against_final(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return _mse_along_batch(preds, targets_final)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.mse(preds, targets).mean()

    def __repr__(self) -> None:
        s = f"MSEModule"
        return s


class MultiTermLossFunction(torch.nn.Module):
    def __init__(self, pred_idx: int, scale_factor: float = 1.0) -> None:
        super().__init__()

        self.pred_idx = pred_idx
        self.scale_factor = scale_factor

    def psnr(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return psnr(preds[:, self.pred_idx], targets[:, self.pred_idx])

    def relative_l2_error(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return relative_l2_error(preds[:, self.pred_idx], targets[:, self.pred_idx])

    def mse(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return _mse_along_batch(preds[:, self.pred_idx], targets[:, self.pred_idx])

    def relative_l2_error_against_final(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return relative_l2_error(preds[:, self.pred_idx], targets_final)

    def mse_against_final(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return _mse_along_batch(preds[:, self.pred_idx], targets_final)

    def psnr_against_final(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        return psnr(preds[:, self.pred_idx], targets_final)

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        targets_final: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Given predictions at a series of frequencies, and targets at the same series of
        frequencies, compute the loss function:

        L(preds, targets) = \sum_i ((N_freqs - i) ** scale_factor) * mse(preds_freq_i, targets_freq_i)

        Args:
            preds (torch.Tensor): Has shape (batch, N_freqs, N_theta, N_rho)
            targets (torch.Tensor): Has shape (batch, N_freqs, N_theta, N_rho)

        Returns:
            torch.Tensor: Scalar loss value
        """
        loss = 0
        for i in range(preds.shape[1]):
            loss += (preds.shape[1] - i) ** self.scale_factor * mse(
                preds[:, i], targets[:, i]
            ).mean()
        return loss

    def __repr__(self) -> None:
        s = f"MultiTermLossFunction with pred_idx: {self.pred_idx} and scale_factor: {self.scale_factor}"
        return s
