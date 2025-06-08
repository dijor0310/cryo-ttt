import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as L
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from torch.nn.functional import sigmoid
import wandb
from utils.ddw_subtomos import reassemble_subtomos
import torchvision.transforms.functional as FT
from torchmetrics.functional import precision, recall

from torchmetrics.functional.classification import binary_confusion_matrix, binary_precision, binary_recall
import mrcfile
from torch.nn.functional import binary_cross_entropy_with_logits
from models import UNet3D

def normalize_min_max(image):
    return (image - image.min()) * 255 / (image.max() - image.min())

def dice_from_conf_matrix(conf_matrix):
    return 2 * conf_matrix[1, 1] / (2 * conf_matrix[1, 1] + conf_matrix[0, 1] + conf_matrix[1, 0])

def precision_from_conf_matrix(conf_matrix: torch.Tensor):
    return conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

def recall_from_conf_matrix(conf_matrix: torch.Tensor):
    return conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])


class Denoiseg(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = self.config.learning_rate

        self.model = UNet3D(
            out_channels=2,
            depth=config.depth,
            initial_features=config.initial_features,
            decoder_dropout=config.decoder_dropout,
            encoder_dropout=config.encoder_dropout,
            BN=config.BN,
            elu=config.elu,
            final_activation=None,
        )

        # self.criterion = DiceLoss(sigmoid=True)
        # self.criterion = DiceFocalLoss(sigmoid=True)
        self.criterion = DiceCELoss(sigmoid=True, lambda_ce=self.config.lambda_ce)
        self.reconstruction_loss = nn.MSELoss(reduction="none")

        self.dice_loss = DiceLoss(sigmoid=True)
        self.dice_score = DiceMetric()

        self.val_start_coords = []
        self.val_preds = []
        self.val_gt = torch.Tensor(mrcfile.read(self.config.val_gt))[None, None, ...]

    def normalize(self, x):

        mean = x.mean(dim=(-4, -3, -2, -1), keepdim=True)
        std = x.std(dim=(-4, -3, -2 , -1), keepdim=True) + 1e-6
        return (x - mean) / std, mean, std

    def denormalize(self, x, mean, std):
        return std * x + mean
    
    def masked_rec_loss(self, pred, target, mask):
        masked_loss = self.reconstruction_loss(pred, target) * mask

        return masked_loss.sum() / mask.sum()

    def training_step(self, batch, batch_idx):
        x, y_out = batch["image"], batch["label"]
        x_out, mask = batch["unmasked_image"], batch["mask"]

        # ### Normalize
        # x, mean, std = self.normalize(x)
        # ###

        y_hat, x_hat = self.model(x)

        # ### Denormalize
        # x_hat = self.denormalize(x_hat, mean, std)
        # ###

        denoising_loss = self.masked_rec_loss(x_hat, x_out, mask)
        seg_loss = self.criterion(y_hat, y_out)
        loss = seg_loss + denoising_loss
        acc = ((sigmoid(y_hat) > 0.5).int() == y_out).sum() / torch.numel(y_hat)

        self.log_dict(
            {
                "train/loss": loss,
                "train/accuracy": acc,
                "train/dice_loss": self.dice_loss(y_hat, y_out),
                "train/mse_loss": denoising_loss,
                "train/ce_loss": binary_cross_entropy_with_logits(y_hat, y_out.float()),
            },
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )


        return loss

    def validation_step(self, batch, batch_idx):
        x, y_out = batch["image"], batch["label"]
        x_out, mask = batch["unmasked_image"], batch["mask"]


        # ### Normalize
        # x, mean, std = self.normalize(x)
        # ###

        y_hat, x_hat = self.model(x)

        # ### Denormalize
        # x_hat = self.denormalize(x_hat, mean, std)
        # ###

        denoising_loss = self.masked_rec_loss(x_hat, x_out, mask)
        seg_loss = self.criterion(y_hat, y_out)
        loss = seg_loss + denoising_loss
        acc = ((sigmoid(y_hat) > 0.5).int() == y_out).sum() / torch.numel(y_hat)

        self.log_dict(
            {
                "val/loss": loss,
                "val/accuracy": acc,
                "val/dice_loss": self.dice_loss(y_hat, y_out),
                "val/mse_loss": denoising_loss,
                "val/ce_loss": binary_cross_entropy_with_logits(y_hat, y_out.float()),
            },
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )

        if "start_coord" in batch:
            start_coord = batch["start_coord"]
            gathered_start_coords = self.all_gather(start_coord.detach())
            gathered_y_hats = self.all_gather(y_hat.squeeze(1).detach())

            if self.trainer.global_rank == 0:
                # gathered_start_coords = [elem.cpu() for elem in gathered_start_coords]
                # gathered_y_hats = [elem.cpu().sigmoid() for elem in gathered_y_hats]

                self.val_start_coords.extend(torch.unbind(gathered_start_coords.cpu().flatten(end_dim=1), dim=0))
                self.val_preds.extend(torch.unbind(gathered_y_hats.cpu().sigmoid().flatten(end_dim=1), dim=0))

    def on_validation_epoch_end(self):

        if self.trainer.global_rank != 0:
            return
        
        reassembled_pred = reassemble_subtomos(
            subtomos=self.val_preds,
            subtomo_start_coords=self.val_start_coords,
            subtomo_overlap=80,
            crop_to_size=self.val_gt.shape[2:],
        )

        reassembled_pred_binary = (reassembled_pred[None, None, ...] > 0.5).to(torch.uint8)
        dice_score = self.dice_score(reassembled_pred_binary, self.val_gt)

        self.logger.experiment.log({
            "val/macro_dice": dice_score,
            "val/macro_recall": binary_recall(reassembled_pred_binary, self.val_gt),
            "val/macro_precision": binary_precision(reassembled_pred_binary, self.val_gt),
            "val/rsm_pred": wandb.Image(FT.to_pil_image(normalize_min_max(reassembled_pred_binary.squeeze().sum(dim=0)).to(torch.uint8), mode="L")),
            "epoch": self.current_epoch,
        })

        # self.log(
        #     "val/macro_dice",
        #     dice_score,
        #     on_step=False,
        #     on_epoch=True,
        # )

        
        self.val_start_coords = []
        self.val_preds = []
        # return super().on_validation_epoch_end()
    
    def test_step(self, batch, batch_idx):
        x, y = batch["unmasked_image"], batch["label"]
        id = batch["id"]

        # TODO: calculate dice score on entire tomogram, not on subtomograms
        y_hat, x_denoised = self.model(x)
        # score = torchmetrics.functional.dice(y_hat, y)
        score = 1 - self.criterion(y_hat, y)
        self.log("test/dice", score, batch_size=x.shape[0])

        preds = torch.cat(
            (y[0, 0].sum(dim=0).float() * 100, y_hat[0, 0].sum(dim=0)), dim=0
        )
        denoised = torch.cat(
            (x[0, 0, 128].float(), x_denoised[0, 0, 128].float()), dim=0
        )
        wandb.log(
            {"test/pred": wandb.Image(preds, caption=id[0] + " " + str(score.item()))}
        )
        wandb.log({"test/denoised": wandb.Image(denoised, caption=id[0])})

        # raise NotImplementedError

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=self.learning_rate
        # )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=self.config.method.decay_milestones,
        #     gamma=self.config.method.decay_gamma,
        # )
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer, 
        #     base_lr=self.config.base_lr,   # Minimum learning rate
        #     max_lr=self.config.max_lr,     # Maximum learning rate
        #     step_size_up=self.config.step_size_up,  # Number of iterations to go from base_lr to max_lr
        #     mode='triangular', # Policy for cyclic variation
        #     cycle_momentum=False,
        # )
        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.OneCycleLR(
        #         optimizer,
        #         max_lr=self.learning_rate,
        #         total_steps=self.trainer.estimated_stepping_batches,
        #         pct_start=0.1,
        #         anneal_strategy="cos",
        #     ),
        #     "interval": "step",
        #     "frequency": 1
        # }
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer,
        #     lr_lambda=lambda epoch: (1 - epoch / self.trainer.max_epochs) ** 2.5
        # )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.config.gamma_decay
        )


        return [optimizer], [scheduler]
