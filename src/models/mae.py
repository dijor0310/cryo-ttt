import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.losses import DiceLoss

# # Configuration for the model
# @dataclass
# class AutoencoderConfig:
#     in_channels: int = 1
#     base_channels: int = 32
#     depth: int = 4  # Depth of the network (number of levels in encoder/decoder)
#     kernel_size: int = 3
#     stride: int = 2
#     padding: int = 1
#     dropout: float = 0.1
#     mask_ratio: float = 0.3  # Ratio of the input tensor to be masked

# cs = ConfigStore.instance()
# cs.store(name="autoencoder_config", node=AutoencoderConfig)


class Encoder(nn.Module):
    def __init__(
        self, in_channels, base_channels, depth, kernel_size, stride, padding, dropout
    ):
        super().__init__()
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for level in range(depth):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv3d(
                        current_channels,
                        base_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    # nn.LayerNorm([base_channels, 1, 1, 1]),
                    nn.BatchNorm3d(num_features=base_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(
                        base_channels, base_channels, kernel_size, padding=padding
                    ),
                    # nn.LayerNorm([base_channels, 1, 1, 1]),
                    nn.BatchNorm3d(num_features=base_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(p=dropout),
                )
            )
            current_channels = base_channels
            base_channels *= 2

    def forward(self, x):
        encodings = []
        for layer in self.encoders[:-1]:
            x = layer(x)
            x = F.max_pool3d(x, kernel_size=2, stride=2)
            # print(x.shape)
            encodings.append(x)

        x = self.encoders[-1](x)
        return encodings, x


class Decoder(nn.Module):
    def __init__(self, base_channels, depth, kernel_size, stride, padding, dropout):
        super().__init__()
        self.decoders = nn.ModuleList()
        for level in range(depth - 1):
            self.decoders.append(
                nn.Sequential(
                    nn.Conv3d(
                        base_channels * 3 // 2,
                        base_channels // 2,
                        kernel_size,
                        padding=padding,
                    ),
                    # nn.LayerNorm([base_channels // 2, 1, 1, 1]),
                    nn.BatchNorm3d(num_features=base_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(
                        base_channels // 2,
                        base_channels // 2,
                        kernel_size,
                        padding=padding,
                    ),
                    # nn.LayerNorm([base_channels // 2, 1, 1, 1]),
                    nn.BatchNorm3d(num_features=base_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout3d(p=dropout),
                )
            )
            base_channels //= 2

    def forward(self, x, encodings):
        for layer, encoding in zip(self.decoders, reversed(encodings)):
            # print(x.shape, encoding.shape)
            x = torch.cat([x, encoding], dim=1)
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        return x


class MaskedAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            depth=config.depth,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            dropout=config.dropout,
        )
        self.reconstruction_decoder = Decoder(
            base_channels=config.base_channels * (2 ** (config.depth - 1)),
            # base_channels=config.base_channels * (2 ** (config.depth)),
            depth=config.depth,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            dropout=config.dropout,
        )
        self.segmentation_decoder = Decoder(
            base_channels=config.base_channels * (2 ** (config.depth - 1)),
            # base_channels=config.base_channels * (2 ** (config.depth)),
            depth=config.depth,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            dropout=config.dropout,
        )
        self.segmentation_head = nn.Conv3d(config.base_channels, 1, kernel_size=1)
        self.reconstruction_head = nn.Conv3d(
            config.base_channels, config.in_channels, kernel_size=1
        )
        self.mask_ratio = config.mask_ratio

    def apply_mask(self, x):
        mask = torch.rand_like(x) > self.mask_ratio
        return x * mask, mask

    def forward(self, x):
        masked_x, mask = self.apply_mask(x)
        encodings, bottleneck = self.encoder(masked_x)
        reconstruction = self.reconstruction_decoder(bottleneck, encodings)
        segmentation = self.segmentation_decoder(bottleneck, encodings)
        # reconstruction = torch.sigmoid(self.reconstruction_head(reconstruction))
        reconstruction = self.reconstruction_head(reconstruction)
        # segmentation = torch.sigmoid(self.segmentation_head(segmentation))
        segmentation = self.segmentation_head(segmentation)
        return segmentation, reconstruction, mask


class LightningMaskedAutoencoder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = MaskedAutoencoder(config)
        self.config = config
        self.mse_loss = nn.MSELoss(reduction="none")
        self.dice_loss = DiceLoss(sigmoid=True)
        self.current_stage = (
            config.stage
        )  # 1: Pre-train reconstruction, 2: Train segmentation

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = (
            batch["image"],
            batch["label"],
        )  # Assuming input is (x, y), where y is the segmentation ground truth
        segmentation, reconstruction, mask = self.model(x)
        if self.current_stage == 1:
            loss = self.compute_reconstruction_loss(x, reconstruction, mask)
            self.log("train/reconstruction_loss", loss)
        else:
            loss = self.compute_segmentation_loss(segmentation, y)
            self.log("train/segmentation_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        segmentation, reconstruction, mask = self.model(x)
        if self.current_stage == 1:
            loss = self.compute_reconstruction_loss(x, reconstruction, mask)
            self.log("val/reconstruction_loss", loss)
        else:
            loss = self.compute_segmentation_loss(segmentation, y)
            self.log("val/segmentation_loss", loss)

    def compute_reconstruction_loss(self, x, reconstruction, mask):
        # x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        # reconstruction: Reconstructed tensor of the same shape as x
        # mask: Binary mask tensor of the same shape as x,
        # where 1 indicates valid voxels and 0 indicates masked ones
        non_masked_loss = self.mse_loss(reconstruction, x) * mask
        loss = non_masked_loss.sum() / mask.sum()
        return loss

    def compute_segmentation_loss(self, segmentation, targets):
        # segmentation: Predicted segmentation of shape
        # (batch_size, 1, depth, height, width)
        # targets: Ground truth segmentation of the same shape
        loss = self.dice_loss(segmentation, targets)
        return loss

    def configure_optimizers(self):

        # optimizer = torch.optim.SGD(
        #     self.network.parameters(),
        #     self.config.method.learning_rate,
        #     weight_decay=self.config.method.weight_decay,
        #     momentum=0.99,
        #     nesterov=True
        # )

        return torch.optim.Adam(self.parameters(), lr=self.config.method.learning_rate)
        # return optimizer

    def on_train_epoch_start(self):
        if self.current_stage == 1:
            # Freeze segmentation decoder and head
            for param in self.model.segmentation_decoder.parameters():
                param.requires_grad = False
            for param in self.model.segmentation_head.parameters():
                param.requires_grad = False
            # Unfreeze encoder and reconstruction decoder and head
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            for param in self.model.reconstruction_decoder.parameters():
                param.requires_grad = True
            for param in self.model.reconstruction_head.parameters():
                param.requires_grad = True
        elif self.current_stage == 2:
            # Freeze encoder and reconstruction decoder and head
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.reconstruction_decoder.parameters():
                param.requires_grad = False
            for param in self.model.reconstruction_head.parameters():
                param.requires_grad = False
            # Unfreeze segmentation decoder and head
            for param in self.model.segmentation_decoder.parameters():
                param.requires_grad = True
            for param in self.model.segmentation_head.parameters():
                param.requires_grad = True
