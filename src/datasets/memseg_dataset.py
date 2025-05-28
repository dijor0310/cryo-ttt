import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
from typing import Dict
import numpy as np
from pathlib import Path
import pickle
import random

from datasets.memseg.memseg_augmentation import get_training_transforms, get_training_f2fd_transforms

class MemSegDataset(Dataset):
    def __init__(self, config, is_validation=False, is_test=False):

        self.is_test = is_test
        self.is_validation = is_validation

        if is_test:
            self.root_dir = Path(config.test_data_root_dir)
            self.tomo_names = config.test_tomo_names
            self.config = config
            self.image_filenames = [file for tomo_name in self.tomo_names for file in self.root_dir.joinpath(tomo_name).rglob("*.pkl")]
            return 
        
        self.root_dir = config.root_dir
        self.config = config
        if is_validation:
            self.image_dir = os.path.join(config.root_dir, "imagesVal")
            self.label_dir = os.path.join(config.root_dir, "labelsVal")
            self.transforms = None

        else:
            self.image_dir = os.path.join(config.root_dir, "imagesTr")
            self.label_dir = os.path.join(config.root_dir, "labelsTr")
            self.transforms = get_training_f2fd_transforms(prob_to_one=False)

        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".nii.gz")])
        self.label_filenames = [f.rsplit("_", 1)[0] + ".nii.gz" for f in self.image_filenames]
        
    def __len__(self):
        return len(self.image_filenames)
    
    def load_nii(self, filepath, dtype):
        nii_img = nib.load(filepath)
        return np.array(nii_img.get_fdata(), dtype=dtype)
    
    def __getitem__(self, idx):
        
        if self.is_test:
            return self.get_test_sample(idx)

        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])
        
        image = self.load_nii(img_path, dtype=np.float32)
        label = self.load_nii(label_path, dtype=np.uint8)

        if self.config.normalize_data:
            image = (image - image.mean()) / (image.std() + 1e-8)
                
        sample = {
            "image": image[np.newaxis, ...],
            "label": label[np.newaxis, ...]
        }

        sample = self.get_random_crop(sample)

        if self.transforms:
            sample = self.transforms(sample)
            
        return sample


    def get_test_sample(self, idx):
        file_path = self.image_filenames[idx]
        

        with open(file_path, 'rb') as f:
            sample = pickle.load(f)

        image = sample["subtomo"]
        label = sample["label"]
        start_coord = torch.Tensor(sample["start_coord"])

        if self.config.normalize_data:
            image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Membrain-seg follows x-y-z format
        # image = image.swapaxes(-1, -3)
        # label = label.swapaxes(-1, -3)
        # start_coord[..., [0, 2]] = start_coord[..., [2, 0]]

        return {
            "image": image.unsqueeze(0).to(torch.float32),
            "label": label.unsqueeze(0).to(torch.uint8),
            "id": sample["tomo_name"] + "/" + "_".join(str(start_coord) for start_coord in sample["start_coord"]),
            "start_coord": start_coord.to(torch.int32),
        }

    def get_random_crop(self, idx_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Returns a random crop from the image-label pair.

        Parameters
        ----------
        idx_dict : Dict[str, np.ndarray]
            A dictionary containing an image and its corresponding label.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing a random crop from the image and its corresponding
            label.
        """
        img, label = idx_dict["image"], idx_dict["label"]
        x, y, z = img.shape[1:]

        if x <= self.config.patch_size or y <= self.config.patch_size or z <= self.config.patch_size:
            # pad with 2s on both sides
            pad_x = max(self.config.patch_size - x, 0)
            pad_y = max(self.config.patch_size - y, 0)
            pad_z = max(self.config.patch_size - z, 0)
            img = np.pad(
                img,
                (
                    (0, 0),
                    (pad_x // 2, pad_x // 2),
                    (pad_y // 2, pad_y // 2),
                    (pad_z // 2, pad_z // 2),
                ),
                mode="constant",
                constant_values=0,
            )
            label = np.pad(
                label,
                (
                    (0, 0),
                    (pad_x // 2, pad_x // 2),
                    (pad_y // 2, pad_y // 2),
                    (pad_z // 2, pad_z // 2),
                ),
                mode="constant",
                constant_values=2,
            )
            # make sure there was no rounding issue
            if (
                img.shape[1] < self.config.patch_size
                or img.shape[2] < self.config.patch_size
                or img.shape[3] < self.config.patch_size
            ):
                img = np.pad(
                    img,
                    (
                        (0, 0),
                        (0, max(self.config.patch_size - img.shape[1], 0)),
                        (0, max(self.config.patch_size - img.shape[2], 0)),
                        (0, max(self.config.patch_size - img.shape[3], 0)),
                    ),
                    mode="constant",
                    constant_values=0,
                )
                label = np.pad(
                    label,
                    (
                        (0, 0),
                        (0, max(self.config.patch_size - label.shape[1], 0)),
                        (0, max(self.config.patch_size - label.shape[2], 0)),
                        (0, max(self.config.patch_size - label.shape[3], 0)),
                    ),
                    mode="constant",
                    constant_values=2,
                )
            assert (
                img.shape[1] == self.config.patch_size
                and img.shape[2] == self.config.patch_size
                and img.shape[3] == self.config.patch_size
            ), f"Image shape is {img.shape} instead of {self.config.patch_size}"
            return {"image": img, "label": label}

        x_crop, y_crop, z_crop = self.config.patch_size, self.config.patch_size, self.config.patch_size
        x_start = np.random.randint(0, x - x_crop)
        y_start = np.random.randint(0, y - y_crop)
        z_start = np.random.randint(0, z - z_crop)
        img = img[
            :,
            x_start : x_start + x_crop,
            y_start : y_start + y_crop,
            z_start : z_start + z_crop,
        ]
        label = label[
            :,
            x_start : x_start + x_crop,
            y_start : y_start + y_crop,
            z_start : z_start + z_crop,
        ]

        assert (
            img.shape[1] == self.config.patch_size
            and img.shape[2] == self.config.patch_size
            and img.shape[3] == self.config.patch_size
        ), f"Image shape is {img.shape} instead of {self.config.patch_size}"
        return {"image": img, "label": label}



class MemSegF2FDDataset(Dataset):
    def __init__(self, config, is_validation=False, is_test=False):

        self.is_test = is_test
        if is_test:
            # raise NotImplementedError
            self.root_dir = Path(config.test_data_root_dir)
            self.tomo_names = config.test_tomo_names
            self.config = config
            self.image_filenames = [file for tomo_name in self.tomo_names for file in self.root_dir.joinpath(tomo_name).rglob("*.pkl")]
            return 

        
        self.is_validation = is_validation
        
        self.root_dir = config.root_dir
        self.config = config

        self.image_dir = os.path.join(config.root_dir, "imagesTr")
        self.label_dir = os.path.join(config.root_dir, "labelsTr")

        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".nii.gz") and f.startswith(tuple(config.data_prefix))])
        self.label_filenames = [f.rsplit("_", 1)[0] + ".nii.gz" for f in self.image_filenames]

        # for scaling law!! 
        # num_samples = len(self.image_filenames)
        # portion = int(config.dataset_portion * num_samples)
        # self.image_filenames = self.image_filenames[:portion]
        # self.label_filenames = self.label_filenames[:portion]

        # delimiter = int(0.2 * len(self.image_filenames))
        if self.is_validation:

            self.image_dir = os.path.join(config.root_dir, "imagesVal")
            self.label_dir = os.path.join(config.root_dir, "labelsVal")

            self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".nii.gz") and f.startswith(tuple(config.data_prefix))])
            self.label_filenames = [f.rsplit("_", 1)[0] + ".nii.gz" for f in self.image_filenames]

            # self.image_filenames = self.image_filenames[:delimiter]
            # self.label_filenames = self.label_filenames[:delimiter]
        
            self.transforms = None
        else:
            # self.image_filenames = self.image_filenames[delimiter:]
            # self.label_filenames = self.label_filenames[delimiter:]

            # self.transforms = get_training_transforms(prob_to_one=False)
            self.transforms = get_training_f2fd_transforms(prob_to_one=False)
        
    def __len__(self):
        return len(self.image_filenames)
        
    def __getitem__(self, idx):
        
        if self.is_test:
            return self.get_test_sample(idx)

        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])
        
        image = self.load_nii(img_path, dtype=np.float32)
        label = self.load_nii(label_path, dtype=np.uint8)

        if self.config.normalize_data:
            image = (image - image.mean()) / (image.std() + 1e-8)
                
        sample = {
            "image": image[np.newaxis, ...],
            "label": label[np.newaxis, ...]
        }

        sample = self.get_random_crop(sample)

        if self.transforms:
            sample = self.transforms(sample)


        masked_raw1, masked_raw2 = self.fourier_masking_twice(sample["image"].squeeze())
        masked_raw1, masked_raw2 = torch.Tensor(masked_raw1), torch.Tensor(masked_raw2)

        return sample | {
            "noisy_1": masked_raw1.unsqueeze(0).to(torch.float32),
            "noisy_2": masked_raw2.unsqueeze(0).to(torch.float32)
        }


    def load_nii(self, filepath, dtype):
        nii_img = nib.load(filepath)
        return np.array(nii_img.get_fdata(), dtype=dtype)

    def get_random_crop(self, idx_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Returns a random crop from the image-label pair.

        Parameters
        ----------
        idx_dict : Dict[str, np.ndarray]
            A dictionary containing an image and its corresponding label.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing a random crop from the image and its corresponding
            label.
        """
        img, label = idx_dict["image"], idx_dict["label"]
        x, y, z = img.shape[1:]

        if x <= self.config.patch_size or y <= self.config.patch_size or z <= self.config.patch_size:
            # pad with 2s on both sides
            pad_x = max(self.config.patch_size - x, 0)
            pad_y = max(self.config.patch_size - y, 0)
            pad_z = max(self.config.patch_size - z, 0)
            img = np.pad(
                img,
                (
                    (0, 0),
                    (pad_x // 2, pad_x // 2),
                    (pad_y // 2, pad_y // 2),
                    (pad_z // 2, pad_z // 2),
                ),
                mode="constant",
                constant_values=0,
            )
            label = np.pad(
                label,
                (
                    (0, 0),
                    (pad_x // 2, pad_x // 2),
                    (pad_y // 2, pad_y // 2),
                    (pad_z // 2, pad_z // 2),
                ),
                mode="constant",
                constant_values=2,
            )
            # make sure there was no rounding issue
            if (
                img.shape[1] < self.config.patch_size
                or img.shape[2] < self.config.patch_size
                or img.shape[3] < self.config.patch_size
            ):
                img = np.pad(
                    img,
                    (
                        (0, 0),
                        (0, max(self.config.patch_size - img.shape[1], 0)),
                        (0, max(self.config.patch_size - img.shape[2], 0)),
                        (0, max(self.config.patch_size - img.shape[3], 0)),
                    ),
                    mode="constant",
                    constant_values=0,
                )
                label = np.pad(
                    label,
                    (
                        (0, 0),
                        (0, max(self.config.patch_size - label.shape[1], 0)),
                        (0, max(self.config.patch_size - label.shape[2], 0)),
                        (0, max(self.config.patch_size - label.shape[3], 0)),
                    ),
                    mode="constant",
                    constant_values=2,
                )
            assert (
                img.shape[1] == self.config.patch_size
                and img.shape[2] == self.config.patch_size
                and img.shape[3] == self.config.patch_size
            ), f"Image shape is {img.shape} instead of {self.config.patch_size}"
            return {"image": img, "label": label}

        x_crop, y_crop, z_crop = self.config.patch_size, self.config.patch_size, self.config.patch_size
        x_start = np.random.randint(0, x - x_crop)
        y_start = np.random.randint(0, y - y_crop)
        z_start = np.random.randint(0, z - z_crop)
        img = img[
            :,
            x_start : x_start + x_crop,
            y_start : y_start + y_crop,
            z_start : z_start + z_crop,
        ]
        label = label[
            :,
            x_start : x_start + x_crop,
            y_start : y_start + y_crop,
            z_start : z_start + z_crop,
        ]

        assert (
            img.shape[1] == self.config.patch_size
            and img.shape[2] == self.config.patch_size
            and img.shape[3] == self.config.patch_size
        ), f"Image shape is {img.shape} instead of {self.config.patch_size}"
        return {"image": img, "label": label}

    def fourier_masking_twice(self, patch):
        """Applies Fourier-based perturbations to create paired noisy patches."""
        fft_patch = np.fft.rfftn(patch)
        fft_patch = np.fft.fftshift(fft_patch, axes=(-3, -2))
        
        fft_patch1 = self.mask(fft_patch)
        fft_patch2 = self.mask(fft_patch)

        # Inverse FFT
        ifft_patch1 = np.fft.irfftn(np.fft.ifftshift(fft_patch1, axes=(-3, -2)), s=patch.shape)
        ifft_patch2 = np.fft.irfftn(np.fft.ifftshift(fft_patch2, axes=(-3, -2)), s=patch.shape)
        
        return ifft_patch1.astype(np.float32), ifft_patch2.astype(np.float32)

    def mask(self, fft_patch):
        # Patch-based Bernoulli Masking (8x8x8 patches)
        size = fft_patch.shape[0]
        patch_size = 8
        mask_shape = (size // patch_size, size // patch_size, 1 + (size // 2 + 1) // patch_size)
        patch_mask = (np.random.rand(*mask_shape) > self.config.bernoulli_mask_ratio).astype(np.float32)
        
        # Expand mask to match fft_patch shape
        bernoulli_mask = patch_mask.repeat(patch_size, axis=0)
        bernoulli_mask = bernoulli_mask.repeat(patch_size, axis=1)
        bernoulli_mask = bernoulli_mask.repeat(patch_size, axis=2)
        bernoulli_mask = bernoulli_mask[:size, :size, :size // 2 + 1]  # Ensure matching shape
        
        # fft_masked = fft_patch * bernoulli_mask
        
        # Random Phase Inversion
        phase_flip = (np.random.rand(*fft_patch.real.shape) < self.config.phase_inversion_ratio).astype(np.float32)
        fft_patch_inverted = fft_patch * ((-1) ** phase_flip)
        
        # Spherical Mask to keep low frequencies
        center = size // 2
        # radius = random.randint(int(0.05 * size), int(0.1 * size))
        radius = random.randint(int(self.config.min_mask_radius * size), int(self.config.max_mask_radius * size))
        x, y, z = np.meshgrid(torch.arange(size), torch.arange(size), torch.arange(size // 2 + 1), indexing='ij')
        mask = ((x - center) ** 2 + (y - center) ** 2 + (z) ** 2) < (radius ** 2)
        
        mask = mask.astype(np.float32)

        overall_mask = np.logical_or(mask, bernoulli_mask)
        # fft_masked = torch.where(mask, fft_masked, torch.zeros_like(fft_masked))
        return fft_patch_inverted * overall_mask.astype(np.float32)

    def get_test_sample(self, idx):
        file_path = self.image_filenames[idx]
        

        with open(file_path, 'rb') as f:
            sample = pickle.load(f)

        image = sample["subtomo"]
        label = sample["label"]
        start_coord = torch.Tensor(sample["start_coord"])

        if self.config.normalize_data:
            image = (image - image.mean()) / (image.std() + 1e-8)
        
        masked_raw1, masked_raw2 = self.fourier_masking_twice(image.squeeze())
        masked_raw1, masked_raw2 = torch.Tensor(masked_raw1), torch.Tensor(masked_raw2)

        return {
            "image": image.unsqueeze(0).to(torch.float32),
            "label": label.unsqueeze(0).to(torch.uint8),
            "noisy_1": masked_raw1.unsqueeze(0).to(torch.float32),
            "noisy_2": masked_raw2.unsqueeze(0).to(torch.float32),
            "id": sample["tomo_name"] + "/" + "_".join(str(start_coord) for start_coord in sample["start_coord"]),
            "start_coord": start_coord.to(torch.int32),
        }
