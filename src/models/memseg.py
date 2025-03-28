import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as L
from monai.losses import DiceLoss, DiceCELoss, MaskedLoss, MaskedDiceLoss
from monai.metrics import DiceMetric
from torch.nn.functional import sigmoid
import wandb
from utils.ddw_subtomos import reassemble_subtomos
import torchvision.transforms.functional as FT
from torchmetrics.functional import precision, recall

from torchmetrics.functional.classification import binary_confusion_matrix, binary_precision, binary_recall
import mrcfile
from torch.nn.functional import binary_cross_entropy_with_logits

from models.losses import IgnoreLabelDiceCELoss

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

    def bilinear_upsampler(self, in_channels, out_channels):
        module = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear"),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        return module

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
                        self._conv_block_GN_decoder(
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
                self.bilinear_upsampler(n_features_decode[level], n_features_decode[level + 1])
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
        # return x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)
        return x


def normalize_min_max(image):

    return (image - image.min()) * 255 / (image.max() - image.min())

def dice_from_conf_matrix(conf_matrix):
    return 2 * conf_matrix[1, 1] / (2 * conf_matrix[1, 1] + conf_matrix[0, 1] + conf_matrix[1, 0])

def precision_from_conf_matrix(conf_matrix: torch.Tensor):
    return conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])

def recall_from_conf_matrix(conf_matrix: torch.Tensor):
    return conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])


class MemSeg(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.learning_rate = self.config.learning_rate

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

        # self.criterion = DiceCELoss(sigmoid=True, lambda_ce=self.config.lambda_ce)
        # self.reconstruction_loss = nn.MSELoss(reduction="none")
        self.criterion = IgnoreLabelDiceCELoss(
            ignore_label=2,
            reduction='mean',
        )

        # self.dice_loss = MaskedLoss(DiceLoss, reduction='mean')
        self.dice_loss = MaskedDiceLoss(sigmoid=True)
        self.dice_score = DiceMetric()

        self.val_start_coords = []
        self.val_preds = []
        self.val_gt = torch.Tensor(mrcfile.read(self.config.val_gt))[None, None, ...]

        self.val_gt = self.val_gt.swapaxes(-1, -3)

    
    def training_step(self, batch, batch_idx):
        x, y_out = batch["image"], batch["label"]
        batch_size = x.shape[0]

        y_hat = self.model(x)

        loss, bce_loss, dice_loss = self.criterion(y_hat, y_out)
        self.log(
            "train/loss",
            loss,
            batch_size=x.shape[0],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        mask = y_out != 2.0

        acc = (((sigmoid(y_hat) > 0.5).int() == y_out) * mask).sum() / mask.sum()
        self.log(
            "train/accuracy",
            acc,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # # Compute Dice loss separately for each batch element
        # dice_loss = torch.zeros((batch_size,), device=y_hat.device)

        # # TODO (Diyor): I believe this can be unrolled ??
        # for batch_idx in range(y_hat.shape[0]):
        #     dice_loss[batch_idx] = self.dice_loss(
        #         y_hat[batch_idx].unsqueeze(0),
        #         y_out[batch_idx].unsqueeze(0),
        #         mask[batch_idx].unsqueeze(0),
        #     )
        # dice_loss_1 = self.dice_loss(y_hat, y_out, mask)
        self.log(
            "train/dice_loss",
            # dice_loss_1, 
            # dice_loss.mean(),
            dice_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # dice_loss = torch.zeros((batch_size, ), device=y_out.device)

        # # TODO (Diyor): I believe this can be unrolled ??
        # for batch_idx in range(y_out.shape[0]):
        #     dice_loss[batch_idx] = self.dice_loss(
        #         y_hat[batch_idx].unsqueeze(0),
        #         y_out[batch_idx].unsqueeze(0),
        #         mask[batch_idx].unsqueeze(0),
        #     )

        # dice_loss = dice_loss.mean()

        # self.log(
        #     "train/dice_loss_backup",
        #     dice_loss, 
        #     # dice_loss.mean(),
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=batch_size,
        #     sync_dist=True,
        # )

        # assert dice_loss == dice_loss_1, f"1: {dice_loss_1}  0: {dice_loss}"
        self.log(
            "train/ce_loss",
            bce_loss, 
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )


        return loss

    def validation_step(self, batch, batch_idx):
        x, y_out = batch["image"], batch["label"]

        batch_size = x.shape[0]

        y_hat = self.model(x)

        loss, bce_loss, dice_loss = self.criterion(y_hat, y_out)
        self.log(
            "val/loss",
            loss,
            batch_size=x.shape[0],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        mask = y_out != 2.0

        acc = (((sigmoid(y_hat) > 0.5).int() == y_out) * mask).sum() / mask.sum()
        self.log(
            "val/accuracy",
            acc,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # # Compute Dice loss separately for each batch element
        # dice_loss = torch.zeros((batch_size,), device=y_hat.device)

        # # TODO (Diyor): I believe this can be unrolled ??
        # for batch_idx in range(y_hat.shape[0]):
        #     dice_loss[batch_idx] = self.dice_loss(
        #         y_hat[batch_idx].unsqueeze(0),
        #         y_out[batch_idx].unsqueeze(0),
        #         mask[batch_idx].unsqueeze(0),
        #     )

        self.log(
            "val/dice_loss",
            # seg_loss,
            # dice_loss.mean(),
            dice_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.log(
            "val/ce_loss",
            bce_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        
        if "start_coord" in batch:
            start_coord = batch["start_coord"]
            gathered_start_coords = self.all_gather(start_coord.detach())
            gathered_y_hats = self.all_gather(y_hat.squeeze(1).detach())

            if self.trainer.global_rank == 0:

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
        
        self.val_start_coords = []
        self.val_preds = []

    def on_test_start(self):
        if self.trainer.global_rank != 0:
            return
        
        self.test_preds, self.test_start_coords = [], []

        self.test_gt = torch.Tensor(mrcfile.read(self.config.test_entire_gt))[None, None, ...]
        self.test_gt = self.test_gt.swapaxes(-1, -3)

        # TODO ttt before testing (no in this model)

    def test_step(self, batch, batch_idx):        
        x = batch["image"]

        y_hat = self.model(x)


        if "start_coord" in batch:
            start_coord = batch["start_coord"]
            gathered_start_coords = self.all_gather(start_coord.detach())
            gathered_y_hats = self.all_gather(y_hat.squeeze(1).detach())

            if self.trainer.global_rank == 0:

                self.test_start_coords.extend(torch.unbind(gathered_start_coords.cpu().flatten(end_dim=1), dim=0))
                self.test_preds.extend(torch.unbind(gathered_y_hats.cpu().sigmoid().flatten(end_dim=1), dim=0))


    def on_test_epoch_end(self):
        
        if self.trainer.global_rank != 0:
            return
        
        reassembled_pred = reassemble_subtomos(
            subtomos=self.test_preds,
            subtomo_start_coords=self.test_start_coords,
            subtomo_overlap=80,
            crop_to_size=self.test_gt.shape[2:],
        )

        reassembled_pred_binary = (reassembled_pred[None, None, ...] > 0.5).to(torch.uint8)
        dice_score = self.dice_score(reassembled_pred_binary, self.test_gt)

        self.logger.experiment.log({
            "test/macro_dice": dice_score,
            "test/macro_recall": binary_recall(reassembled_pred_binary, self.test_gt),
            "test/macro_precision": binary_precision(reassembled_pred_binary, self.test_gt),
            "test/rsm_pred": wandb.Image(FT.to_pil_image(normalize_min_max(reassembled_pred_binary.squeeze().sum(dim=-1)).to(torch.uint8), mode="L")),
            "epoch": self.current_epoch,
        })

        # self.log(
        #     "val/macro_dice",
        #     dice_score,
        #     on_step=False,
        #     on_epoch=True,
        # )

        
        # self.test_start_coords = []
        # self.val_preds = []
        # return super().on_validation_epoch_end()


    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=self.learning_rate
        # )
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
