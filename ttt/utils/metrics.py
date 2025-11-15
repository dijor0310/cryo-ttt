import torch
from torchmetrics import Metric


class GlobalDiceMetric(Metric):
    def __init__(self, threshold=0.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        # States will be summed across devices if distributed
        self.add_state("global_tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("global_fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("global_fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        # If logits, apply sigmoid and threshold them
        preds = preds > self.threshold
        # Flatten tensors for pixel-wise computation
        preds = preds.int().view(-1)
        target = target.int().view(-1)
        self.global_tp += torch.sum((preds == 1) & (target == 1)).float()
        self.global_fp += torch.sum((preds == 1) & (target == 0)).float()
        self.global_fn += torch.sum((preds == 0) & (target == 1)).float()

    def compute(self) -> torch.Tensor:
        # Compute the Dice score with a smoothing term to avoid division by zero.
        dice = (2 * self.global_tp) / (
            2 * self.global_tp + self.global_fp + self.global_fn + 1e-8
        )
        return dice
