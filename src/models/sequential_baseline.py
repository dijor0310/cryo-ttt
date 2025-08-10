import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as L
from monai.losses import DiceLoss, DiceCELoss, MaskedLoss, MaskedDiceLoss
from monai.metrics import DiceMetric
from torch.nn.functional import sigmoid
import wandb
from utils.ddw_subtomos import reassemble_subtomos
from utils.metrics import GlobalDiceMetric
import torchvision.transforms.functional as FT
from torchmetrics.functional import precision, recall

from torchmetrics.functional.classification import binary_confusion_matrix, binary_precision, binary_recall
from torchmetrics import Dice, Precision, Recall
import mrcfile
from torch.nn.functional import binary_cross_entropy_with_logits

from models.losses import IgnoreLabelDiceCELoss
from models.memseg import MemDenoiseg

from monai.networks.nets import DynUNet
from monai.inferers import SlidingWindowInferer

from datasets.transforms import F2FDMaskingTransform
from monai.transforms import Compose, ToTensor
from models.unet3d import UNet3D
import nibabel as nib
from pathlib import Path
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present



class MemDenoisegTTTSubsetSequential(MemDenoiseg):
    def __init__(self, config):
        super().__init__(config)
        self.global_dice_score = GlobalDiceMetric()
        
        self.seg_model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            kernel_size=self.config.dynunet.kernel_size,
            strides=self.config.dynunet.strides,
            upsample_kernel_size=self.config.dynunet.upsample_kernel_size,
            filters=self.config.dynunet.filters,
            res_block=self.config.dynunet.res_block,
        )

        # state_dict = torch.load("/mnt/hdd_pool_zion/userdata/diyor/ttt_ckpt/memseg-spinach-sequential-on-denoised-tomos-1a1c9/epoch=986-val/dice_loss=0.23.ckpt")
        # consume_prefix_in_state_dict_if_present(state_dict["state_dict"], prefix="model.")
        # self.seg_model.load_state_dict(state_dict["state_dict"])

        configure_model(self.seg_model)

    def forward(self, batch, is_val=False):
        x, y_out = batch["image"], batch["label"]
        x_input, x_target = batch["noisy_1"], batch["noisy_2"]
        batch_size = x.shape[0]
        if not self.config.masked_inference and is_val:
            out = self.model(x)
        else:
            out = self.model(x_input)

        x_hat, y_hat = torch.unbind(out, dim=1)
        x_hat, y_hat = x_hat.unsqueeze(1), y_hat.unsqueeze(1)

        loss, bce_loss, dice_loss = self.criterion(y_hat, y_out)
        rec_loss = self.masked_rec_loss(x_hat, x_target, mask=(y_out != 2))
        loss = loss + rec_loss

        acc = self.masked_accuracy(y_hat, y_out, (y_out != 2.0))

        if is_val:
            # self.global_dice_score.update(y_hat, y_out)

            slice_to_log = (y_out[0].squeeze() == 1).sum(dim=(-1, -2)).argmax()
            self.logger.experiment.log(
                {
                    "val/target": wandb.Image(x_target[0, 0, slice_to_log]),
                    "val/tomo": wandb.Image(x[0, 0, slice_to_log]),
                    "val/denoised": wandb.Image(x_hat[0, 0, slice_to_log]),
                    # "val/label": wandb.Image(y_out[0, 0, slice_to_log]),
                    # "val/predict": wandb.Image(FT.to_pil_image(((y_hat[0, 0, slice_to_log].sigmoid() > 0.5).int() * 255).to(torch.uint8), mode="L"))
                }
            )
            # pass

        return {
            "metrics": {
                # "val/loss" if is_val else "train/loss": loss,
                # "val/ce_loss" if is_val else "train/ce_loss": bce_loss,
                # "val/dice_loss" if is_val else "train/dice_loss": dice_loss,
                "val/mse_loss" if is_val else "train/mse_loss": rec_loss,
                # "val/accuracy" if is_val else "train/accuracy": acc,
            },
            "x_hat": x_hat,
            "y_hat": y_hat,
            "batch_size": batch_size,
            "y_out": y_out,
        }
    
    def training_step(self, batch, batch_idx):

        out = self(batch)
        self.log_dict(
            out["metrics"],
            on_step=False,
            on_epoch=True,
            batch_size=out["batch_size"],
            sync_dist=True,
        )
        return out["metrics"]["train/mse_loss"]
    
    def validation_step(self, batch, batch_idx):
        configure_model(self.seg_model)

        out = self(batch, is_val=True)
        out_2 = self.seg_model(out["x_hat"])

        _, y_hat_2 = torch.unbind(out_2, dim=1)
        y_hat_2 = y_hat_2.unsqueeze(1)

        loss, bce_loss, dice_loss = self.criterion(y_hat_2, out["y_out"])
        # rec_loss = self.masked_rec_loss(x_hat, x_target, mask=(y_out != 2))
        # loss = loss + rec_loss
        acc = self.masked_accuracy(y_hat_2, out["y_out"], (out["y_out"] != 2.0))

        self.global_dice_score.update(y_hat_2, out["y_out"])

        out["metrics"] = out["metrics"] | {
            "val/loss": loss,
            "val/ce_loss": bce_loss,
            "val/dice_loss": dice_loss,
            "val/accuracy": acc,
        }

        self.log_dict(
            out["metrics"],
            on_step=False,
            on_epoch=True,
            batch_size=out["batch_size"],
            sync_dist=True,
        )

        slice_to_log = (out["y_out"][0].squeeze() == 1).sum(dim=(-1, -2)).argmax()
        self.logger.experiment.log(
            {
                # "val/target": wandb.Image(x_target[0, 0, slice_to_log]),
                # "val/tomo": wandb.Image(x[0, 0, slice_to_log]),
                # "val/denoised": wandb.Image(x_hat[0, 0, slice_to_log]),
                "val/label": wandb.Image(out["y_out"][0, 0, slice_to_log]),
                "val/predict": wandb.Image(FT.to_pil_image(((y_hat_2[0, 0, slice_to_log].sigmoid() > 0.5).int() * 255).to(torch.uint8), mode="L"))
            }
        )


    def on_validation_epoch_end(self):
        self.logger.experiment.log({
            "val/global_dice": self.global_dice_score.compute(),
        })
        self.global_dice_score.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        if self.config.use_sgd:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
            )

        return optimizer


def configure_model(model):

    model.eval()

    for param in model.parameters():
        param.requires_grad = False