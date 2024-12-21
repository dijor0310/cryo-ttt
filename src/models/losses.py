import torch
from monai.losses import DiceLoss, MaskedLoss
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    sigmoid,
)


class IgnoreLabelDiceCELoss(_Loss):
    """
    Mix of Dice & Cross-entropy loss, adding ignore labels.

    Parameters
    ----------
    ignore_label : int
        The label to ignore when calculating the loss.
    reduction : str, optional
        Specifies the reduction to apply to the output, by default "mean".
    lambda_dice : float, optional
        The weight for the Dice loss, by default 1.0.
    lambda_ce : float, optional
        The weight for the Cross-Entropy loss, by default 1.0.
    kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        ignore_label: int,
        reduction: str = "none",
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        self.ignore_label = ignore_label
        self.dice_loss = MaskedLoss(DiceLoss, reduction=reduction, **kwargs)
        self.reduction = reduction
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of model outputs.
        target : torch.Tensor
            Tensor of target labels.

        Returns
        -------
        torch.Tensor
            The calculated loss.
        """
        # Create a mask to ignore the specified label in the target
        orig_data = data.clone()
        data = sigmoid(data)
        mask = target != self.ignore_label

        # Compute the cross entropy loss while ignoring the ignore_label
        target_comp = target.clone()
        target_comp[target == self.ignore_label] = 0
        target_tensor = torch.tensor(target_comp, dtype=data.dtype, device=data.device)

        bce_loss = binary_cross_entropy_with_logits(
            orig_data, target_tensor, reduction="none"
        )
        bce_loss[~mask] = 0.0
        # TODO: Check if this is correct: I adjusted the loss to be
        # computed per batch element
        bce_loss = torch.sum(bce_loss, dim=(1, 2, 3, 4)) / torch.sum(
            mask, dim=(1, 2, 3, 4)
        )
        # Compute Dice loss separately for each batch element
        dice_loss = torch.zeros_like(bce_loss)
        for batch_idx in range(data.shape[0]):
            dice_loss[batch_idx] = self.dice_loss(
                data[batch_idx].unsqueeze(0),
                target[batch_idx].unsqueeze(0),
                mask[batch_idx].unsqueeze(0),
            )

        # Combine the Dice and Cross Entropy losses
        combined_loss = self.lambda_dice * dice_loss + self.lambda_ce * bce_loss
        if self.reduction == "mean":
            combined_loss = combined_loss.mean()
        elif self.reduction == "sum":
            combined_loss = combined_loss.sum()
        elif self.reduction == "none":
            return combined_loss
        else:
            raise ValueError(
                f"Invalid reduction type {self.reduction}. "
                "Valid options are 'mean' and 'sum'."
            )
        return combined_loss
