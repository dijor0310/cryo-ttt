import numpy as np
import torch
from monai.transforms import Transform
from typing import Dict


class F2FDMaskingTransform(Transform):
    def __init__(
        self,
        bernoulli_mask_ratio: float = 0.5,
        phase_inversion_ratio: float = 0.1,
        min_mask_radius: float = 0.05,
        max_mask_radius: float = 0.1,
    ):
        super().__init__()
        self.bernoulli_mask_ratio = bernoulli_mask_ratio
        self.phase_inversion_ratio = phase_inversion_ratio
        self.min_mask_radius = min_mask_radius
        self.max_mask_radius = max_mask_radius

    def __call__(self, patch: Dict):
        fft_patch = torch.fft.rfftn(patch, s=patch.shape[2:], dim=(-3, -2, -1))
        fft_patch = torch.fft.fftshift(fft_patch, dim=(-3, -2))

        # print(fft_patch.shape)
        masked_fft_patch = self.mask(fft_patch)

        # Inverse FFT
        ifft_patch = torch.fft.irfftn(
            torch.fft.ifftshift(masked_fft_patch, dim=(-3, -2)),
            dim=(-3, -2, -1),
            s=patch.shape[2:],
        )

        return ifft_patch

    def mask(self, fft_patch):
        size = fft_patch.shape[-2]
        patch_size = 8
        mask_shape = (
            size // patch_size,
            size // patch_size,
            1 + (size // 2 + 1) // patch_size,
        )
        # print(mask_shape)
        patch_mask = (np.random.rand(*mask_shape) > self.bernoulli_mask_ratio).astype(
            np.float32
        )
        # patch_mask = torch.tensor((np.random.rand(*mask_shape) > self.bernoulli_mask_ratio), dtype=torch.float, device=fft_patch.device)

        # Expand mask to match fft_patch shape
        bernoulli_mask = patch_mask.repeat(patch_size, axis=0)
        bernoulli_mask = bernoulli_mask.repeat(patch_size, axis=1)
        bernoulli_mask = bernoulli_mask.repeat(patch_size, axis=2)
        bernoulli_mask = bernoulli_mask[
            :size, :size, : size // 2 + 1
        ]  # Ensure matching shape
        bernoulli_mask = torch.tensor(
            bernoulli_mask, dtype=torch.float, device=fft_patch.device
        )

        # Random Phase Inversion
        phase_flip = (
            np.random.rand(*fft_patch.real.shape) < self.phase_inversion_ratio
        ).astype(np.float32)
        phase_flip = torch.tensor(
            phase_flip, dtype=torch.float, device=fft_patch.device
        )
        fft_patch_inverted = fft_patch * ((-1) ** phase_flip)

        # Spherical Mask to keep low frequencies
        center = size // 2
        radius = np.random.randint(
            int(self.min_mask_radius * size), int(self.max_mask_radius * size)
        )
        x, y, z = np.meshgrid(
            np.arange(size), np.arange(size), np.arange(size // 2 + 1), indexing="ij"
        )
        mask = ((x - center) ** 2 + (y - center) ** 2 + (z) ** 2) < (radius**2)
        mask = torch.tensor(mask, dtype=torch.float, device=fft_patch.device)

        overall_mask = torch.logical_or(mask, bernoulli_mask)

        # print(fft_patch_inverted.shape, overall_mask.shape)

        return fft_patch_inverted * overall_mask[None, None, ...].repeat(
            fft_patch.shape[0], 1, 1, 1, 1
        )
