import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as L
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.nn.functional import sigmoid
import wandb

# import tensors.actions as actions


def crop_tensor(input_array: np.array, shape_to_crop: tuple) -> np.array:
    """
    Function from A. Kreshuk to crop tensors of order 3, starting always from
    the origin.
    :param input_array: the input np.array image
    :param shape_to_crop: a tuple (cz, cy, cx), where each entry corresponds
    to the size of the  cropped region along each axis.
    :return: np.array of size (cz, cy, cx)
    """
    input_shape = input_array.shape
    assert all(
        ish >= csh for ish, csh in zip(input_shape, shape_to_crop)
    ), "Input shape must be larger equal crop shape"
    # get the difference between the shapes
    shape_diff = tuple((ish - csh) // 2 for ish, csh in zip(input_shape, shape_to_crop))
    # calculate the crop
    crop = tuple(slice(sd, sh - sd) for sd, sh in zip(shape_diff, input_shape))
    return input_array[crop]


class UNet3D(nn.Module):
    """UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
      depth: depth of the u-net (= number of encoder / decoder levels)
      initial_features: number of features after first encoder
    """

    def _conv_block_GN_encoder(self, in_channels, out_channels, emb_dim):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.GroupNorm(4, out_channels),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.GroupNorm(4, out_channels),
            nn.Dropout(p=self.encoder_dropout),
        )

    def _conv_block_GN_decoder(self, in_channels, out_channels, emb_dim):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.GroupNorm(4, out_channels),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.GroupNorm(4, out_channels),
            nn.Dropout(p=self.encoder_dropout),
        )

    def _conv_block_IN_encoder(self, in_channels, out_channels, emb_dim):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout(p=self.encoder_dropout),
        )

    def _conv_block_IN_decoder(self, in_channels, out_channels, emb_dim):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm3d(out_channels),
            nn.Dropout(p=self.encoder_dropout),
        )

    def _conv_block_LN_encoder(self, in_channels, out_channels, emb_dim):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm((emb_dim, emb_dim, emb_dim), elementwise_affine=False),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm((emb_dim, emb_dim, emb_dim), elementwise_affine=False),
            nn.Dropout(p=self.encoder_dropout),
        )

    def _conv_block_LN_decoder(self, in_channels, out_channels, emb_dim):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm((emb_dim, emb_dim, emb_dim), elementwise_affine=False),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm((emb_dim, emb_dim, emb_dim), elementwise_affine=False),
            nn.Dropout(p=self.encoder_dropout),
        )

    def _conv_block_BN_encoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Dropout(p=self.encoder_dropout),
        )

    def _conv_block_BN_encoder_elu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Dropout(p=self.encoder_dropout),
        )

    def _conv_block_BN_decoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Dropout(p=self.decoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Dropout(p=self.decoder_dropout),
        )

    def _conv_block_decoder_BN_elu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Dropout(p=self.decoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm3d(num_features=out_channels),
            nn.Dropout(p=self.decoder_dropout),
        )

    def _conv_block_encoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.encoder_dropout),
        )

    def _conv_block_decoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.decoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p=self.decoder_dropout),
        )

    def _conv_block_encoder_elu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(p=self.encoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(p=self.encoder_dropout),
        )

    def _conv_block_decoder_elu(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(p=self.decoder_dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Dropout(p=self.decoder_dropout),
        )

    # upsampling via transposed 3d convolutions
    def _upsampler(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        depth=4,
        initial_features=16,
        decoder_dropout=0.1,
        encoder_dropout=0.1,
        # BN: bool = False,
        BN: str = "batch_norm",
        elu=False,
        final_activation=None,
    ):
        super().__init__()

        # Temporary !!! TODO
        box_size = 256

        self.depth = depth
        self.decoder_dropout = decoder_dropout
        self.encoder_dropout = encoder_dropout
        self.non_linearity = nn.ELU() if elu else nn.ReLU
        self.batch_norm = BN

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(
                final_activation, nn.Module
            ), "Activation must be torch module"

        n_features = [initial_features * 2**level for level in range(self.depth)]
        # modules of the encoder path
        n_features_encode = [in_channels] + n_features
        n_features_base = n_features_encode[-1] * 2

        if not elu:
            if BN == "batch_norm":
                self.encoder = nn.ModuleList(
                    [
                        self._conv_block_BN_encoder(
                            n_features_encode[level], n_features_encode[level + 1]
                        )
                        for level in range(self.depth)
                    ]
                )

                # the base convolution block
                self.base = self._conv_block_encoder(
                    n_features_encode[-1], n_features_base
                )
            elif BN == "layer_norm":
                self.encoder = nn.ModuleList(
                    [
                        self._conv_block_LN_encoder(
                            n_features_encode[level],
                            n_features_encode[level + 1],
                            box_size // (2**level),
                        )
                        for level in range(self.depth)
                    ]
                )

                # the base convolution block
                self.base = self._conv_block_encoder(
                    n_features_encode[-1], n_features_base
                )
            elif BN == "instance_norm":
                self.encoder = nn.ModuleList(
                    [
                        self._conv_block_IN_encoder(
                            n_features_encode[level],
                            n_features_encode[level + 1],
                            box_size // (2**level),
                        )
                        for level in range(self.depth)
                    ]
                )

                # the base convolution block
                self.base = self._conv_block_encoder(
                    n_features_encode[-1], n_features_base
                )
            elif BN == "group_norm":
                self.encoder = nn.ModuleList(
                    [
                        self._conv_block_GN_encoder(
                            n_features_encode[level],
                            n_features_encode[level + 1],
                            box_size // (2**level),
                        )
                        for level in range(self.depth)
                    ]
                )

                # the base convolution block
                self.base = self._conv_block_encoder(
                    n_features_encode[-1], n_features_base
                )
            else:
                self.encoder = nn.ModuleList(
                    [
                        self._conv_block_encoder(
                            n_features_encode[level], n_features_encode[level + 1]
                        )
                        for level in range(self.depth)
                    ]
                )

                # the base convolution block
                self.base = self._conv_block_encoder(
                    n_features_encode[-1], n_features_base
                )
        else:
            if BN == "batch_norm":
                self.encoder = nn.ModuleList(
                    [
                        self._conv_block_BN_encoder_elu(
                            n_features_encode[level], n_features_encode[level + 1]
                        )
                        for level in range(self.depth)
                    ]
                )

                # the base convolution block
                self.base = self._conv_block_encoder_elu(
                    n_features_encode[-1], n_features_base
                )
            # elif BN == "layer_norm":
            #     self.encoder = nn.ModuleList(
            #         [
            #             self._conv_block_BN_encoder_elu(
            #                 n_features_encode[level], n_features_encode[level + 1]
            #             )
            #             for level in range(self.depth)
            #         ]
            #     )

            #     # the base convolution block
            #     self.base = self._conv_block_encoder_elu(
            #         n_features_encode[-1], n_features_base
            #     )
            else:
                self.encoder = nn.ModuleList(
                    [
                        self._conv_block_encoder_elu(
                            n_features_encode[level], n_features_encode[level + 1]
                        )
                        for level in range(self.depth)
                    ]
                )

                # the base convolution block
                self.base = self._conv_block_encoder_elu(
                    n_features_encode[-1], n_features_base
                )
        # modules of the decoder path
        n_features_decode = [n_features_base] + n_features[::-1]
        # print("decoder:", n_features_decode)
        if not elu:
            if BN == "batch_norm":
                self.decoder = nn.ModuleList(
                    [
                        self._conv_block_BN_decoder(
                            n_features_decode[level], n_features_decode[level + 1]
                        )
                        for level in range(self.depth)
                    ]
                )
            elif BN == "layer_norm":
                self.decoder = nn.ModuleList(
                    [
                        self._conv_block_LN_decoder(
                            n_features_decode[level],
                            n_features_decode[level + 1],
                            box_size // (2 ** (self.depth - level - 1)),
                        )
                        for level in range(self.depth)
                    ]
                )
            elif BN == "instance_norm":
                self.decoder = nn.ModuleList(
                    [
                        self._conv_block_IN_decoder(
                            n_features_decode[level],
                            n_features_decode[level + 1],
                            box_size // (2 ** (self.depth - level - 1)),
                        )
                        for level in range(self.depth)
                    ]
                )
            elif BN == "group_norm":
                self.decoder = nn.ModuleList(
                    [
                        self._conv_block_IN_decoder(
                            n_features_decode[level],
                            n_features_decode[level + 1],
                            box_size // (2 ** (self.depth - level - 1)),
                        )
                        for level in range(self.depth)
                    ]
                )
            else:
                self.decoder = nn.ModuleList(
                    [
                        self._conv_block_decoder(
                            n_features_decode[level], n_features_decode[level + 1]
                        )
                        for level in range(self.depth)
                    ]
                )

        else:
            if BN == "batch_norm":
                self.decoder = nn.ModuleList(
                    [
                        self._conv_block_decoder_BN_elu(
                            n_features_decode[level], n_features_decode[level + 1]
                        )
                        for level in range(self.depth)
                    ]
                )
            # elif BN == "layer_norm":
            #     self.decoder = nn.ModuleList(
            #         [
            #             self._conv_block_decoder_BN_elu(
            #                 n_features_decode[level], n_features_decode[level + 1]
            #             )
            #             for level in range(self.depth)
            #         ]
            #     )
            else:
                self.decoder = nn.ModuleList(
                    [
                        self._conv_block_decoder_elu(
                            n_features_decode[level], n_features_decode[level + 1]
                        )
                        for level in range(self.depth)
                    ]
                )

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList([nn.MaxPool3d(2) for _ in range(self.depth)])

        # the upsampling layers
        self.upsamplers = nn.ModuleList(
            [
                self._upsampler(n_features_decode[level], n_features_decode[level + 1])
                for level in range(self.depth)
            ]
        )
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = nn.Conv3d(initial_features, out_channels, 1)
        self.activation = final_activation

    # crop the `from_encoder` tensor and concatenate both
    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)

    def forward(self, input_tensor):
        x = input_tensor
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](self._crop_and_concat(x, encoder_out[level]))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)


class Denoiseg(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

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

        self.criterion = DiceLoss(sigmoid=True)
        # self.criterion = DiceFocalLoss(sigmoid=True)
        self.reconstruction_loss = nn.MSELoss(reduction="none")

        self.dice_loss = DiceLoss(sigmoid=True)
        self.dice_score = DiceMetric()

    def masked_rec_loss(self, pred, target, mask):
        masked_loss = self.reconstruction_loss(pred, target) * mask

        return masked_loss.sum() / mask.sum()

    def training_step(self, batch, batch_idx):
        x, y_out = batch["image"], batch["label"]
        x_out, mask = batch["unmasked_image"], batch["mask"]
        batch_size = x.shape[0]

        y_hat, x_hat = self.model(x)

        denoising_loss = self.masked_rec_loss(x_hat, x_out, mask)
        seg_loss = self.criterion(y_hat, y_out)
        loss = seg_loss + denoising_loss
        self.log(
            "train/loss",
            loss,
            batch_size=x.shape[0],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        acc = ((sigmoid(y_hat) > 0.5).int() == y_out).sum() / torch.numel(y_hat)
        self.log(
            "train/accuracy",
            acc,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/dice_loss",
            seg_loss,
            # self.dice_loss(y_hat, y_out),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "train/mse_loss",
            denoising_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_out = batch["image"], batch["label"]
        x_out, mask = batch["unmasked_image"], batch["mask"]
        batch_size = x.shape[0]

        y_hat, x_hat = self.model(x)

        denoising_loss = self.masked_rec_loss(x_hat, x_out, mask)
        seg_loss = self.criterion(y_hat, y_out)
        loss = seg_loss + denoising_loss
        self.log(
            "val/loss",
            loss,
            batch_size=x.shape[0],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        acc = ((sigmoid(y_hat) > 0.5).int() == y_out).sum() / torch.numel(y_hat)
        self.log(
            "val/accuracy",
            acc,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/dice_loss",
            seg_loss,
            # self.dice_loss(y_hat, y_out),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/mse_loss",
            denoising_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch["unmasked_image"], batch["label"]
        id = batch["id"]

        # TODO: calculate dice score on entire tomogram, not on subtomograms
        y_hat, x_denoised = self.model(x)
        # score = torchmetrics.functional.dice(y_hat, y)
        score = 1 - self.dice_loss(y_hat, y)
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
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.method.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config.method.decay_milestones,
            gamma=self.config.method.decay_gamma,
        )

        return [optimizer], [scheduler]
