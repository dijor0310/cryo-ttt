import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as L
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import torchmetrics
from torch.nn.functional import sigmoid

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


# def crop_window(input_array: np.array, shape_to_crop: tuple or list,
#                 window_corner: tuple or list):
#     """
#     Function from A. Kreshuk to crop tensors of order 3, starting always
#     from a given corner.
#     :param input_array: the input np.array image
#     :param shape_to_crop: a tuple (cz, cy, cx), where each entry corresponds
#     to the size of the  cropped region along each axis.
#     :param window_corner: point from where the window will be cropped.
#     :return: np.array of size (cz, cy, cx)
#     """
#     input_shape = input_array.shape
#     assert all(ish >= csh for ish, csh in zip(input_shape, shape_to_crop)), \
#         "Input shape must be larger equal crop shape"
#     # get the difference between the shapes
#     crop = tuple(slice(wc, wc + csh)
#                  for wc, csh in zip(window_corner, shape_to_crop))
#     # print(crop)
#     return input_array[crop]


# def crop_window_around_point(input_array: np.array, crop_shape: tuple or list,
#                              window_center: tuple or list) -> np.array:
#     # The window center is not in tom_coordinates, it is (z, y, x)
#     input_shape = input_array.shape
#     assert all(ish - csh // 2 - center >= 0 for ish, csh, center in
#                zip(input_shape, crop_shape, window_center)), \
#         "Input shape must be larger or equal than crop shape"
#     assert all(center - csh // 2 >= 0 for csh, center in
#                zip(crop_shape, window_center)), \
#         "Input shape around window center must be larger equal than crop shape"
#     # get the difference between the shapes
#     crop = tuple(slice(center - csh // 2, center + csh // 2)
#                  for csh, center in zip(crop_shape, window_center))
#     return input_array[crop]


class UNet3D(nn.Module):
    """UNet implementation
    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
      depth: depth of the u-net (= number of encoder / decoder levels)
      initial_features: number of features after first encoder
    """

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

    def forward(self, input_tensor, y):

        if y is None:
            y = torch.full_like(input_tensor, 0.5)
        # x = input_tensor
        x = torch.cat([input_tensor, y], dim=1)
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
        return x


class UNet3D_Lightning_ITTT(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = UNet3D(
            in_channels=2,
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
        # self.criterion = IgnoreLabelDiceCELoss(
        #     ignore_label=2, reduction="mean", lambda_dice=1, lambda_ce=1
        # )

        self.dice_score = DiceMetric()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.rotation_cross_entropy = nn.CrossEntropyLoss()

    # def js_div(p, q):
    #     """Function that computes distance between two predictions"""
    #     m = 0.5 * (p + q)
    #     return 0.5 * (F.kl_div(torch.log(p), m, reduction='batchmean') +
    #                 F.kl_div(torch.log(q), m, reduction='batchmean'))

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        batch["id"]
        batch_size = x.shape[0]

        mask = torch.rand(1) > 0.8
        y_in = y if mask else None

        y_hat_0 = self.model(x, y=y_in)
        # y_hat_1 = self.model(x, y=y_hat_0)

        loss = self.criterion(y_hat_0, y)  # + self.criterion(y_hat_1, y)
        self.log(
            "train/loss",
            loss,
            batch_size=batch_size,
            on_step=False,
            on_epoch=True,
        )

        acc = ((sigmoid(y_hat_0) > 0.5).int() == y).sum() / torch.numel(y_hat_0)
        self.log(
            "train/accuracy", acc, on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log(
            "train/dice_loss",
            self.dice_loss(y_hat_0, y),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        batch["id"]
        batch_size = x.shape[0]

        torch.rand(1) > 0.8
        # y_in = y if mask else None

        y_hat_0 = self.model(x, y=None)
        y_hat_1 = self.model(x, y=y_hat_0)

        loss = self.criterion(y_hat_0, y)
        self.log("val/loss", loss, batch_size=batch_size, on_step=False, on_epoch=True)

        acc = ((sigmoid(y_hat_0) > 0.5).int() == y).sum() / torch.numel(y_hat_0)
        self.log(
            "val/accuracy", acc, on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log(
            "val/dice_loss",
            self.dice_loss(y_hat_0, y),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/dice_loss_1",
            self.dice_loss(y_hat_1, y),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/dice_zigzag",
            self.dice_loss(y_hat_0, y_hat_1),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        # TODO: calculate dice score on entire tomogram, not on subtomograms
        y_hat, _ = self.model(x)
        score = torchmetrics.functional.dice(y_hat, y)
        self.log("test/dice", score, batch_size=x.shape[0])

    def configure_optimizers(self):
        # return super().configure_optimizers()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.method.learning_rate
        )

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer,
        #     milestones=self.config.method.decay_milestones,
        #     gamma=self.config.method.decay_gamma,
        # )

        # lr_scheduler = {
        #     "scheduler": scheduler,
        #     "name": "scheduler",
        # }
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.config.method.base_lr,
            max_lr=self.config.method.max_lr,
            step_size_up=self.config.method.step_size,
            step_size_down=self.config.method.step_size,
            cycle_momentum=False,
        )
        return [optimizer], [scheduler]
