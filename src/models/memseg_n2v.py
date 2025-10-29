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

from monai.networks.nets import DynUNet
from monai.inferers import SlidingWindowInferer

from datasets.transforms import F2FDMaskingTransform
from monai.transforms import Compose, ToTensor
from models.unet3d import UNet3D

def normalize_min_max(image):
    return (image - image.min()) * 255 / (image.max() - image.min())

def dice_from_conf_matrix(conf_matrix):
    return 2 * conf_matrix[1, 1] / (2 * conf_matrix[1, 1] + conf_matrix[0, 1] + conf_matrix[1, 0])

def precision_from_conf_matrix(conf_matrix: torch.Tensor):
    return conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

def recall_from_conf_matrix(conf_matrix: torch.Tensor):
    return conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

def standardize(tensor: torch.Tensor):
    print("Mean intensity: ", tensor.mean())
    print("Std intensity: ", tensor.std())
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)


class MemDenoiseg(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = self.config.learning_rate

        if self.config.dynunet.enable:
            self.model = DynUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=2,
                kernel_size=self.config.dynunet.kernel_size,
                strides=self.config.dynunet.strides,
                upsample_kernel_size=self.config.dynunet.upsample_kernel_size,
                filters=self.config.dynunet.filters,
                res_block=self.config.dynunet.res_block,
                norm_name=(self.config.dynunet.norm_name, {'affine': True}),
            )
        else:
            self.model = UNet3D(
                out_channels=1,
                depth=config.depth,
                initial_features=config.initial_features,
                decoder_dropout=config.decoder_dropout,
                encoder_dropout=config.encoder_dropout,
                BN=config.BN,
                elu=config.elu,
                final_activation=None,
            )

        self.reconstruction_loss = nn.MSELoss(reduction='none')
        self.criterion = IgnoreLabelDiceCELoss(
            ignore_label=2,
            reduction='mean',
        )

        self.dice_loss = MaskedDiceLoss(sigmoid=True)
        self.dice_score = DiceMetric()

        self.global_dice_score = GlobalDiceMetric()

        self.val_start_coords = []
        self.val_preds = []
        # self.val_gt = torch.Tensor(mrcfile.read(self.config.val_gt))[None, None, ...]
        self.inferer = SlidingWindowInferer(
            roi_size=(config.patch_size, ) * 3,
            sw_batch_size=config.test_batch_size,
            overlap=0.5,
            progress=True,
            mode="gaussian",
            device=torch.device("cpu"), # this is device for stitching (reassembling) 
        )


    def masked_rec_loss(self, pred, target, mask):
        masked_loss = self.reconstruction_loss(pred, target) * mask
        return masked_loss.sum() / mask.sum()
    
    def masked_accuracy(self, pred, target, mask):
        return (((sigmoid(pred) > 0.5).int() == target) * mask.int()).sum() / mask.sum()

    def training_step(self, batch, batch_idx):
        x, y_out = batch["image"], batch["label"]
        
        x_input, x_target = batch["masked_image"], batch["image"]
        mask = batch["mask"]
        out = self.model(x_input)

        x_hat, y_hat = torch.unbind(out, dim=1)

        x_hat, y_hat = x_hat.unsqueeze(1), y_hat.unsqueeze(1)

        loss, bce_loss, dice_loss = self.criterion(y_hat, y_out)
        rec_loss = self.masked_rec_loss(x_hat, x_target, mask=mask)

        loss = loss + rec_loss
        acc = self.masked_accuracy(y_hat, y_out, (y_out != 2.0))

        self.log(
            "train/loss",
            loss,
            batch_size=x.shape[0],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        self.log_dict(
            {
                "train/loss": loss,
                "train/accuracy": acc,
                "train/dice_loss": dice_loss,
                "train/ce_loss": bce_loss,
                "train/mse_loss": rec_loss,
            },
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_out = batch["image"], batch["label"]
        
        x_input, x_target = batch["masked_image"], batch["image"]
        mask = batch["mask"]
        batch_size = x.shape[0]

        out = self.model(x_input)

        x_hat, y_hat = torch.unbind(out, dim=1)

        x_hat, y_hat = x_hat.unsqueeze(1), y_hat.unsqueeze(1)

        loss, bce_loss, dice_loss = self.criterion(y_hat, y_out)
        rec_loss = self.masked_rec_loss(x_hat, x_target, mask=mask)

        loss = loss + rec_loss
        acc = self.masked_accuracy(y_hat, y_out, (y_out != 2.0))

        self.global_dice_score.update(y_hat, y_out)

        self.log_dict(
            {
                "val/loss": loss,
                "val/accuracy": acc,
                "val/dice_loss": dice_loss,
                "val/ce_loss": bce_loss,
                "val/mse_loss": rec_loss,
            },
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )

        slice_to_log = (y_out[0].squeeze() == 1).sum(dim=(-1, -2)).argmax()
        self.logger.experiment.log(
            {
                "val/target": wandb.Image(x_target[0, 0, slice_to_log]),
                "val/tomo": wandb.Image(x[0, 0, slice_to_log]),
                "val/denoised": wandb.Image(x_hat[0, 0, slice_to_log]),
                "val/label": wandb.Image(y_out[0, 0, slice_to_log]),
                "val/predict": wandb.Image(FT.to_pil_image(((y_hat[0, 0, slice_to_log].sigmoid() > 0.5).int() * 255).to(torch.uint8), mode="L"))
            }
        )

    def on_validation_epoch_end(self):

        self.logger.experiment.log({
            "val/global_dice": self.global_dice_score.compute(),
        })
        self.global_dice_score.reset()

    def test_step(self, batch, batch_idx):

        if self.config.monai_inference and batch_idx != 0:
            return     

        if self.config.monai_inference:
            self.monai_test_pred = self.inferer(
                inputs=self.test_tomo.to(device=torch.device(self.trainer.local_rank)),
                network=self.model,
            )
            return
                
        x_hats, y_hats = [], []
        for key in self.config.inference_key:
            x = batch[key]
            out = self.model(x)
            x_hat, y_hat = torch.unbind(out, dim=1)
            x_hat, y_hat = x_hat.unsqueeze(1), y_hat.unsqueeze(1)

            x_hats.append(x_hat)
            y_hats.append(y_hat)

        x_hat = torch.stack(x_hats, dim=0).mean(dim=0)
        y_hat = torch.stack(y_hats, dim=0).mean(dim=0)
        
    def on_test_epoch_end(self):

        if self.config.monai_inference:
            self.monai_test_pred = (self.monai_test_pred[:, 1].unsqueeze(1) > 0.0).int()
            dice_score = self.dice_score(self.monai_test_pred, self.test_gt)

            self.logger.experiment.log({
                "test/macro_dice": dice_score,
                "test/macro_recall": binary_recall(self.monai_test_pred, self.test_gt),
                "test/macro_precision": binary_precision(self.monai_test_pred, self.test_gt),
                "test/rsm_pred": wandb.Image(FT.to_pil_image(normalize_min_max(self.monai_test_pred.squeeze().sum(dim=0)).to(torch.uint8), mode="L")),
                "epoch": self.current_epoch,
            })
            return

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.config.gamma_decay
        )


        return [optimizer], [scheduler]


class MemDenoisegTTTSubset(MemDenoiseg):
    def __init__(self, config):
        super().__init__(config)
        self.global_dice_score = GlobalDiceMetric()

    def forward(self, batch, is_val=False):
        # x, y_out = batch["image"], batch["label"]
        # x_input, x_target = batch["noisy_1"], batch["noisy_2"]
        x, y_out = batch["image"], batch["label"]
        
        x_input, x_target = batch["masked_image"], batch["image"]
        mask = batch["mask"]

        batch_size = x.shape[0]
        if not self.config.masked_inference and is_val:
            out = self.model(x)
        else:
            out = self.model(x_input)

        x_hat, y_hat = torch.unbind(out, dim=1)
        x_hat, y_hat = x_hat.unsqueeze(1), y_hat.unsqueeze(1)

        loss, bce_loss, dice_loss = self.criterion(y_hat, y_out)
        rec_loss = self.masked_rec_loss(x_hat, x_target, mask=mask)
        loss = loss + rec_loss

        acc = self.masked_accuracy(y_hat, y_out, (y_out != 2.0))

        if is_val:
            self.global_dice_score.update(y_hat, y_out)

            slice_to_log = (y_out[0].squeeze() == 1).sum(dim=(-1, -2)).argmax()
            self.logger.experiment.log(
                {
                    "val/target": wandb.Image(x_target[0, 0, slice_to_log]),
                    "val/tomo": wandb.Image(x[0, 0, slice_to_log]),
                    "val/denoised": wandb.Image(x_hat[0, 0, slice_to_log]),
                    "val/label": wandb.Image(y_out[0, 0, slice_to_log]),
                    "val/predict": wandb.Image(FT.to_pil_image(((y_hat[0, 0, slice_to_log].sigmoid() > 0.5).int() * 255).to(torch.uint8), mode="L"))
                }
            )

        return {
            "metrics": {
                "val/loss" if is_val else "train/loss": loss,
                "val/ce_loss" if is_val else "train/ce_loss": bce_loss,
                "val/dice_loss" if is_val else "train/dice_loss": dice_loss,
                "val/mse_loss" if is_val else "train/mse_loss": rec_loss,
                "val/accuracy" if is_val else "train/accuracy": acc,
            },
            "x_hat": x_hat,
            "y_hat": y_hat,
            "batch_size": batch_size
        }
    
    def training_step(self, batch, batch_idx):
        if self.config.optimize_only_bn:
            self.configure_model()
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
        out = self(batch, is_val=True)
        self.log_dict(
            out["metrics"],
            on_step=False,
            on_epoch=True,
            batch_size=out["batch_size"],
            sync_dist=True,
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

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def configure_model_val(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.eval()
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm3d):
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
