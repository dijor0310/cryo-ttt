import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_msssim import ssim
from torchmetrics.functional import peak_signal_noise_ratio

# This is just a copy from the original implementation in:
# https://github.com/NVIDIA/partialconv/tree/master/models
from .partialconv3d import PartialConv3d
from monai.losses import DiceLoss
import wandb
import torchvision.transforms.functional as FT
import torch.nn.functional as F

from models.losses import self2selfLoss_noMask


class Denoising_3DUNet(pl.LightningModule):
    def __init__(self, config):
        """Expected input: [B, C, S, S, S] where B the batch size, C input channels and S the subtomo length.
        The data values are expected to be standardized and [0, 1] scaled.
        """

        super().__init__()
        self.config = config
        self.loss_fn = nn.MSELoss()
        self.segmentation_loss = DiceLoss(sigmoid=True)

        self.lr = self.config.learning_rate
        self.n_features = self.config.n_features
        self.p = self.config.dropout
        self.n_bernoulli_samples = self.config.n_bernoulli_samples
        self.in_channels = 1
        self.save_hyperparameters()

        # Encoder blocks
        self.EB0 = PartialConv3d(
            self.in_channels, self.n_features, kernel_size=3, padding=1
        )
        self.EB1 = self.encoder_block()
        self.EB2 = self.encoder_block()
        self.EB3 = self.encoder_block()
        self.EB4 = self.encoder_block()
        self.EB5 = self.encoder_block()
        self.EB6 = self.encoder_block_bottom()

        # Upsampling
        self.up54 = nn.Upsample(scale_factor=2)
        self.up43 = nn.Upsample(scale_factor=2)
        self.up32 = nn.Upsample(scale_factor=2)
        self.up21 = nn.Upsample(scale_factor=2)
        self.up10 = nn.Upsample(scale_factor=2)

        # decoder blocks
        self.DB5 = self.decoder_block(2 * self.n_features, 2 * self.n_features)
        self.DB4 = self.decoder_block(3 * self.n_features, 2 * self.n_features)
        self.DB3 = self.decoder_block(3 * self.n_features, 2 * self.n_features)
        self.DB2 = self.decoder_block(3 * self.n_features, 2 * self.n_features)
        # self.DB1 = self.decoder_block_top()
        self.DB1 = self.decoder_block_denoiseg()

        return

    def forward(self, x: torch.Tensor):
        "Input tensor of shape [batch_size, channels, tomo_side, tomo_side, tomo_side]"
        ##### ENCODER #####
        e0 = self.EB0(x)  # no downsampling, n_features = 48
        e1 = self.EB1(e0)  # downsamples 1/2
        e2 = self.EB2(e1)  # 1/4
        e3 = self.EB3(e2)  # 1/8
        e4 = self.EB4(e3)  # 1/16
        e5 = self.EB5(e4)  # 1/32
        e6 = self.EB6(e5)  # only Pconv and LReLu
        # for debugging
        # print('EB0 (no downsampling):', e0.shape)
        # print('EB1:', e1.shape)
        # print('EB2:', e2.shape)
        # print('EB3:', e3.shape)
        # print('EB4:', e4.shape)
        # print('EB5:', e5.shape)
        # print('EB6: (no downsampling)', e6.shape)

        ##### DECODER #####
        d5 = self.up54(e6)  # 1/16
        d5 = torch.concat([d5, e4], axis=1)  # 1/16, n_freatures = 96
        d5 = self.DB5(d5)  # 1/16

        d4 = self.up43(d5)  # 1/8
        d4 = torch.concat([d4, e3], axis=1)  # 1/8 n_features = 144
        d4 = self.DB4(d4)  # 1/8 n_features = 96

        d3 = self.up32(d4)  # 1/4
        d3 = torch.concat([d3, e2], axis=1)  # 1/4
        d3 = self.DB3(d3)  # 1/4

        d2 = self.up21(d3)  # 1/2
        d2 = torch.concat([d2, e1], axis=1)  # 1/2
        d2 = self.DB2(d2)  # 1/2

        d1 = self.up10(d2)
        d1 = torch.concat([d1, x], axis=1)
        x = self.DB1(d1)

        return x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)

    def encoder_block(self):
        layer = nn.Sequential(
            PartialConv3d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        return layer

    def encoder_block_bottom(self):
        layer = nn.Sequential(
            PartialConv3d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        return layer

    def decoder_block(self, n_features_in, n_features_out):
        layer = nn.Sequential(
            nn.Dropout(self.p),
            nn.Conv3d(n_features_in, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv3d(n_features_out, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        return layer

    def decoder_block_top(self):
        layer = nn.Sequential(
            nn.Dropout(self.p),
            nn.Conv3d(
                2 * self.n_features + self.in_channels, 64, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv3d(32, self.in_channels, kernel_size=3, padding=1),
            # This is in the original implementation paper, but here it doesn't help.
            # It forces data to be (almost) positive, while tomogram data is close to normal around zero.
            # nn.LeakyReLU(0.1)
        )
        return layer
    
    def decoder_block_denoiseg(self):
        layer = nn.Sequential(
            nn.Dropout(self.p),
            nn.Conv3d(
                2 * self.n_features + self.in_channels, 64, kernel_size=3, padding=1
            ),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.p),
            nn.Conv3d(32, self.in_channels + 1, kernel_size=3, padding=1),
            # This is in the original implementation paper, but here it doesn't help.
            # It forces data to be (almost) positive, while tomogram data is close to normal around zero.
            # nn.LeakyReLU(0.1)
        )
        return layer


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr
        )

        return [optimizer]

    def training_step(self, batch, batch_idx):
        bernoulli_subtomo = batch["subtomo"]
        target = batch["target"]
        # bernoulli_mask = batch["bernoulli_mask"]
        gt_subtomo = batch["gt_subtomo"]
        gt_membrane = batch["gt_membrane"]

        pred_segm, pred = self(bernoulli_subtomo)
        segm_loss = self.segmentation_loss(pred_segm, gt_membrane)
        loss = F.mse_loss(pred, target) + segm_loss

        # loss = denoising_loss #+ segm_loss

        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # sync_dist=True,
        )

        self.log(
            "train/dice_loss",
            segm_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # sync_dist=True
        )

        self.log(
            "train/mse_loss",
            F.mse_loss(pred, target),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # sync_dist=True
        )

        # self.log
        self.logger.experiment.log({
            "train/input": wandb.Image((bernoulli_subtomo[0, :, 80])),
            "train/target": wandb.Image((target[0, :, 80])),
            "train/gt_subtomo": wandb.Image(gt_subtomo[0, :, 80]),
            "train/seg_pred": wandb.Image(FT.to_pil_image(pred_segm[0].sum(dim=1).to(torch.uint8), mode="L")),
            "train/seg_gt": wandb.Image(FT.to_pil_image(gt_membrane[0].sum(dim=1).to(torch.uint8), mode="L")),
            "train/denoised": wandb.Image(pred[0, :, 80]),
            "epoch": self.current_epoch,
        })

        # if gt_subtomo is not None:
        #     bernoulliBatch_subtomo = self.batch2bernoulliBatch(bernoulli_subtomo)
        #     bernoulliBatch_pred = self.batch2bernoulliBatch(pred)
        #     bernoulliBatch_gt_subtomo = self.batch2bernoulliBatch(gt_subtomo)
        #     baseline_ssim, baseline_psnr = self.ssim_psnr_monitoring(
        #         bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
        #     )
        #     monitor_ssim, monitor_psnr = self.ssim_psnr_monitoring(
        #         bernoulliBatch_pred, bernoulliBatch_gt_subtomo
        #     )

        #     self.log(
        #         "ssim/baseline",
        #         baseline_ssim,
        #         on_step=False,
        #         on_epoch=True,
        #         # prog_bar=False,
        #         # sync_dist=True,
        #     )

        #     self.log(
        #         "ssim/predicted",
        #         monitor_ssim,
        #         on_step=False,
        #         on_epoch=True,
        #         # prog_bar=False,
        #         # sync_dist=True,
        #     )

        #     self.log(
        #         "psnr/baseline",
        #         baseline_psnr,
        #         on_step=False,
        #         on_epoch=True,
        #         # prog_bar=False,
        #         # sync_dist=True,
        #     )

        #     self.log(
        #         "psnr/predicted",
        #         monitor_psnr,
        #         on_step=False,
        #         on_epoch=True,
        #         # prog_bar=False,
        #         # sync_dist=True,
        #     )

        # tensorboard = self.logger.experiment
        # tensorboard.add_histogram(
        #     "Intensity distribution", pred.detach().cpu().numpy().flatten()
        # )

        return loss
    

    def validation_step(self, batch):
        bernoulli_subtomo = batch["subtomo"]
        target = batch["target"]
        gt_subtomo = batch["gt_subtomo"]
        gt_membrane = batch["gt_membrane"]

        pred_segm, pred = self(bernoulli_subtomo)
        segm_loss = self.segmentation_loss(pred_segm, gt_membrane)
        denoising_loss = self.loss_fn(pred, target)

        loss = segm_loss + denoising_loss

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/dice_loss",
            segm_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            "val/mse_loss",
            denoising_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )


    def batch2bernoulliBatch(self, subtomo):
        return torch.split(subtomo, self.n_bernoulli_samples)

    def ssim_psnr_monitoring(self, bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo):
        ssim_monitor = 0
        psnr_monitor = 0
        for bBatch_subtomo, bBatch_gt in zip(
            bernoulliBatch_subtomo, bernoulliBatch_gt_subtomo
        ):
            # we first normalize the images
            X = bBatch_subtomo.mean(0)
            X = (X - X.min()) / (X.max() - X.min() + 1e-8)
            Y = bBatch_gt.mean(0)
            Y = (Y - Y.min()) / (Y.max() - Y.min() + 1e-8)

            _ssim, _psnr = float(ssim(X, Y, data_range=1)), float(
                peak_signal_noise_ratio(X, Y, data_range=1)
            )
            ssim_monitor += _ssim
            psnr_monitor += _psnr

        # take the mean wrt batch
        ssim_monitor = ssim_monitor / len(bernoulliBatch_gt_subtomo)
        psnr_monitor = psnr_monitor / len(bernoulliBatch_gt_subtomo)

        return ssim_monitor, psnr_monitor
