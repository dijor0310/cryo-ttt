import torch
from monai.losses import DiceLoss, MaskedLoss
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    sigmoid,
)


class DiceCELoss(_Loss):
    def __init__(
        self,
        reduction: str = "mean",
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        self.dice_loss = DiceLoss(reduction=reduction, **kwargs)
        self.reduction = reduction
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce

    def forward(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = binary_cross_entropy_with_logits(data, target)

        dice_loss = self.dice_loss(data, target)

        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss


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

        # TODO (Diyor): I believe this can be unrolled ??
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
            bce_loss = bce_loss.mean()
            dice_loss = dice_loss.mean()
        elif self.reduction == "sum":
            combined_loss = combined_loss.sum()
            bce_loss = bce_loss.sum()
            dice_loss = dice_loss.mean()
        elif self.reduction == "none":
            return combined_loss, bce_loss, dice_loss
        else:
            raise ValueError(
                f"Invalid reduction type {self.reduction}. "
                "Valid options are 'mean' and 'sum'."
            )
        return combined_loss, bce_loss, dice_loss


class self2selfLoss_noMask(torch.nn.Module):
    def __init__(self, alpha=1e-4):
        super().__init__()
        self.l2 = self2self_L2Loss()
        self.total_variation = TotalVariation()
        self.alpha = alpha

    def forward(self, subtomo_pred, target):
        """
        Tensors of shape: [B, C, S, S, S]
        """
        return self.l2(subtomo_pred, target) + self.alpha * self.total_variation(
            subtomo_pred
        ).mean(0)


class self2self_L2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_wedge, y_hat):
        """
        Tensors of shape: [B, C, S, S, S]
        The loss is only considered in the pixels that are masked from the beginning.
        - y_wedge: (1-bernoulli_mask)*model(bernoulli_subtomo)
        - y_hat: (1-bernoulli_mask)*subtomo

        The loss is the L2 norm across the image, then mean across the batch. The mean across the batch helps to deal with "incomplete"
        batches, which are usually the last ones.
        """
        return torch.linalg.vector_norm(
            y_wedge - y_hat, ord=2, dim=(-4, -3, -2, -1)
        ).mean(0)


def total_variation3D(img: torch.Tensor) -> torch.Tensor:
    r"""Function that computes (Anisotropic) Total Variation according to [1].

    Args:
        img: the input image with shape :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.

    Return:
         a scalar with the computer loss.

    Examples:
        >>> total_variation(torch.ones(3, 4, 4))
        tensor(0.)

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       total_variation_denoising.html>`__.

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

    if len(img.shape) < 4 or len(img.shape) > 5:
        raise ValueError(
            f"Expected input tensor to be of ndim 4 or 5, but got {len(img.shape)}."
        )

    pixel_dif1 = img[..., 1:, :, :] - img[..., :-1, :, :]
    pixel_dif2 = img[..., :, 1:, :] - img[..., :, :-1, :]
    pixel_dif3 = img[..., :, :, 1:] - img[..., :, :, :-1]

    reduce_axes = (-4, -3, -2, -1)

    res = 0
    for pixel_dif in [pixel_dif1, pixel_dif2, pixel_dif3]:
        res += pixel_dif.abs().sum(dim=reduce_axes)

    return res


class TotalVariation(torch.nn.Module):
    r"""Compute the Total Variation according to [1].

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N,)` or scalar.

    Examples:
        >>> tv = TotalVariation()
        >>> output = tv(torch.ones((2, 3, 4, 4), requires_grad=True))
        >>> output.data
        tensor([0., 0.])
        >>> output.sum().backward()  # grad can be implicitly created only for scalar outputs

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """

    def __init__(self):
        super().__init__()

    def forward(self, img) -> torch.Tensor:
        return total_variation3D(img)
