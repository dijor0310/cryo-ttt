import torch
import numpy as np
import torch.nn.functional as F
import torch.fft
from torch.utils.data import Dataset, DataLoader
import mrcfile  # For reading .mrc files
import random
import pickle
from pathlib import Path

class F2FD_Dataset(Dataset):
    def __init__(self, config, is_validation, is_test):
        """
        Args:
            file_paths (list): List of paths to tomogram files (.mrc, .npy, .tiff).
            patch_size (int): Size of the cubic patches extracted.
            num_patches (int): Number of patches to extract per tomogram.
            transform (callable, optional): Optional transform to apply.
        """
        if is_validation:
            self.file_paths = config.val_file_paths
            self.label_paths = config.val_label_paths
        else:
            self.file_paths = config.train_file_paths
            self.label_paths = config.train_label_paths

        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.transform = config.transform
        self.tomograms = [self.load_tomogram(f) for f in self.file_paths]
        self.labels = [self.load_tomogram(f) for f in self.label_paths]
    
    def load_tomogram(self, file_path):
        """Loads a tomogram from an MRC, NPY, or TIFF file."""
        if file_path.endswith('.mrc') or file_path.endswith('.rec'):
            with mrcfile.open(file_path, permissive=True) as mrc:
                return mrc.data.astype(np.float32)
        elif file_path.endswith('.npy'):
            return np.load(file_path).astype(np.float32)
        else:
            raise ValueError("Unsupported file format")
    
    def fourier_masking(self, patch):
        """Applies Fourier-based perturbations to create paired noisy patches."""
        fft_patch = np.fft.rfftn(patch)
        fft_patch = np.fft.fftshift(fft_patch, axes=(-3, -2))
        
        # Patch-based Bernoulli Masking (8x8x8 patches)
        size = patch.shape[0]
        patch_size = 8
        mask_shape = (size // patch_size, size // patch_size, 1 + (size // 2 + 1) // patch_size)
        patch_mask = (np.random.rand(*mask_shape) > 0.5).astype(np.float32)
        
        # Expand mask to match fft_patch shape
        bernoulli_mask = patch_mask.repeat(patch_size, axis=0)
        bernoulli_mask = bernoulli_mask.repeat(patch_size, axis=1)
        bernoulli_mask = bernoulli_mask.repeat(patch_size, axis=2)
        bernoulli_mask = bernoulli_mask[:size, :size, :size // 2 + 1]  # Ensure matching shape
        
        # fft_masked = fft_patch * bernoulli_mask
        
        # Random Phase Inversion
        phase_flip = (np.random.rand(*fft_patch.real.shape) < 0.1).astype(np.float32)
        fft_patch = fft_patch * ((-1) ** phase_flip)
        
        # Spherical Mask to keep low frequencies
        center = size // 2
        radius = random.randint(int(0.05 * size), int(0.1 * size))
        x, y, z = np.meshgrid(torch.arange(size), torch.arange(size), torch.arange(size // 2 + 1), indexing='ij')
        mask = ((x - center) ** 2 + (y - center) ** 2 + (z) ** 2) < (radius ** 2)
        
        mask = mask.astype(np.float32)

        overall_mask = np.logical_or(mask, bernoulli_mask)
        # fft_masked = torch.where(mask, fft_masked, torch.zeros_like(fft_masked))
        fft_patch = fft_patch * overall_mask.astype(np.float32)

        # Inverse FFT
        ifft_patch = np.fft.irfftn(np.fft.ifftshift(fft_patch, axes=(-3, -2)), s=patch.shape)
        
        return ifft_patch.astype(np.float32)
    
    def extract_random_patch(self, tomogram, gt_membrane):
        """Extracts a random cubic patch from a tomogram."""
        d, h, w = tomogram.shape
        pd, ph, pw = self.patch_size, self.patch_size, self.patch_size
        
        d_start = random.randint(0, max(d - pd, 0))
        h_start = random.randint(0, max(h - ph, 0))
        w_start = random.randint(0, max(w - pw, 0))
        
        return tomogram[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw], gt_membrane[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
    
    def pad(self, arr, target_shape):
        """
        Pads a 3D numpy array with zeros to match the target shape.
        
        Parameters:
        arr (numpy.ndarray): Input 3D array.
        target_shape (tuple): Desired shape (depth, height, width).
        
        Returns:
        numpy.ndarray: Padded array with shape equal to target_shape.
        """

        if any(s < a for s, a in zip(target_shape, arr.shape)):
            raise ValueError("Target shape must be greater than or equal to input shape in all dimensions.")
        
        pad_width = [(0, t - a) for t, a in zip(target_shape, arr.shape)]
        
        return np.pad(arr, pad_width, mode='constant', constant_values=0)

    def __len__(self):
        return len(self.file_paths) * self.num_patches
    
    def __getitem__(self, idx):
        """Generates a noisy training pair."""
        # tomogram = random.choice(self.tomograms)
        idx = np.random.randint(len(self.tomograms))
        tomogram = self.tomograms[idx]
        gt_membrane = self.labels[idx]
        
        # print(tomogram.shape)
        patch, gt_membrane = self.extract_random_patch(tomogram, gt_membrane)
        patch = self.pad(patch, (self.patch_size, self.patch_size, self.patch_size))
        gt_membrane = self.pad(gt_membrane, (self.patch_size, self.patch_size, self.patch_size))

        noisy_input = self.fourier_masking(patch)
        noisy_target = self.fourier_masking(patch)
        
        if self.transform:
            noisy_input = self.transform(noisy_input)
            noisy_target = self.transform(noisy_target)
        
        return {
            "subtomo": noisy_input[np.newaxis, ...],
            "target": noisy_target[np.newaxis, ...],
            "gt_membrane": gt_membrane[np.newaxis, ...],
            "gt_subtomo": patch[np.newaxis, ...],
        } # Add channel dim


class F2FD_DatasetV2(Dataset):
    def __init__(self, config, is_validation, is_test):
        """
        Args:
            root_dir (string): Root directory containing tomogram subdirectories.
            tomo_names (list of strings): List of tomogram names to include.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.is_validation = is_validation
        self.is_test = is_test

        if is_validation:
            self.root_dir = Path(config.val_data_root_dir)
            self.tomo_names = config.val_tomo_names
        elif is_test:
            self.root_dir = Path(config.test_data_root_dir)
            self.tomo_names = config.test_tomo_names
        else:
            self.root_dir = Path(config.train_data_root_dir)
            self.tomo_names = config.train_tomo_names

        self.config = config
        self.file_paths = [file for tomo_name in self.tomo_names for file in self.root_dir.joinpath(tomo_name).rglob("*.pkl")]
    
        # self.mask_ratio = config.mask_ratio
        # self.window_size = config.window_size

        torch.set_num_threads(8)

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)

        # masked_raw1 = torch.Tensor(self.fourier_masking(sample["subtomo"]))
        # masked_raw2 = torch.Tensor(self.fourier_masking(sample["subtomo"]))
        masked_raw1, masked_raw2 = self.fourier_masking_twice(sample["subtomo"])
        masked_raw1, masked_raw2 = torch.Tensor(masked_raw1), torch.Tensor(masked_raw2)
        
        return {
            "image": masked_raw1.unsqueeze(0).to(torch.float32),
            "unmasked_image": masked_raw2.unsqueeze(0).to(torch.float32),
            "raw_subtomo": torch.Tensor(sample["subtomo"]).unsqueeze(0).to(torch.float32),
            "mask": torch.ones_like(masked_raw1).unsqueeze(0).to(torch.uint8),
            "label": sample["label"].unsqueeze(0).to(torch.uint8),
            "id": sample["tomo_name"] + "/" + "_".join(str(start_coord) for start_coord in sample["start_coord"]),
            "start_coord": torch.Tensor(sample["start_coord"]).to(torch.int32),
        }
    
    def fourier_masking(self, patch):
        """Applies Fourier-based perturbations to create paired noisy patches."""
        fft_patch = np.fft.rfftn(patch)
        fft_patch = np.fft.fftshift(fft_patch, axes=(-3, -2))
        
        # Patch-based Bernoulli Masking (8x8x8 patches)
        size = patch.shape[0]
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
        fft_patch = fft_patch * ((-1) ** phase_flip)
        
        # Spherical Mask to keep low frequencies
        center = size // 2
        # radius = random.randint(int(0.05 * size), int(0.1 * size))
        radius = random.randint(int(self.config.min_mask_radius * size), int(self.config.max_mask_radius * size))
        x, y, z = np.meshgrid(torch.arange(size), torch.arange(size), torch.arange(size // 2 + 1), indexing='ij')
        mask = ((x - center) ** 2 + (y - center) ** 2 + (z) ** 2) < (radius ** 2)
        
        mask = mask.astype(np.float32)

        overall_mask = np.logical_or(mask, bernoulli_mask)
        # fft_masked = torch.where(mask, fft_masked, torch.zeros_like(fft_masked))
        fft_patch = fft_patch * overall_mask.astype(np.float32)

        # Inverse FFT
        ifft_patch = np.fft.irfftn(np.fft.ifftshift(fft_patch, axes=(-3, -2)), s=patch.shape)
        
        return ifft_patch.astype(np.float32)
    
    def fourier_masking_twice(self, patch):
        """Applies Fourier-based perturbations to create paired noisy patches."""
        fft_patch = np.fft.rfftn(patch)
        fft_patch = np.fft.fftshift(fft_patch, axes=(-3, -2))
        
        # # Patch-based Bernoulli Masking (8x8x8 patches)
        # size = patch.shape[0]
        # patch_size = 8
        # mask_shape = (size // patch_size, size // patch_size, 1 + (size // 2 + 1) // patch_size)
        # patch_mask = (np.random.rand(*mask_shape) > self.config.bernoulli_mask_ratio).astype(np.float32)
        
        # # Expand mask to match fft_patch shape
        # bernoulli_mask = patch_mask.repeat(patch_size, axis=0)
        # bernoulli_mask = bernoulli_mask.repeat(patch_size, axis=1)
        # bernoulli_mask = bernoulli_mask.repeat(patch_size, axis=2)
        # bernoulli_mask = bernoulli_mask[:size, :size, :size // 2 + 1]  # Ensure matching shape
        
        # # fft_masked = fft_patch * bernoulli_mask
        
        # # Random Phase Inversion
        # phase_flip = (np.random.rand(*fft_patch.real.shape) < self.config.phase_inversion_ratio).astype(np.float32)
        # fft_patch = fft_patch * ((-1) ** phase_flip)
        
        # # Spherical Mask to keep low frequencies
        # center = size // 2
        # # radius = random.randint(int(0.05 * size), int(0.1 * size))
        # radius = random.randint(int(self.config.min_mask_radius * size), int(self.config.max_mask_radius * size))
        # x, y, z = np.meshgrid(torch.arange(size), torch.arange(size), torch.arange(size // 2 + 1), indexing='ij')
        # mask = ((x - center) ** 2 + (y - center) ** 2 + (z) ** 2) < (radius ** 2)
        
        # mask = mask.astype(np.float32)

        # overall_mask = np.logical_or(mask, bernoulli_mask)
        # # fft_masked = torch.where(mask, fft_masked, torch.zeros_like(fft_masked))
        # fft_patch = fft_patch * overall_mask.astype(np.float32)

        fft_patch1 = self.mask(fft_patch)
        fft_patch2 = self.mask(fft_patch)

        # if self.is_validation: #or self.is_test:
        #     # Inverse FFT
        #     ifft_patch1 = np.fft.irfftn(np.fft.ifftshift(fft_patch1, axes=(-3, -2)), s=patch.shape)
        #     return ifft_patch1.astype(np.float32), patch

        # else:
        #     fft_patch2 = self.mask(fft_patch)

        #     # Inverse FFT
        #     ifft_patch1 = np.fft.irfftn(np.fft.ifftshift(fft_patch1, axes=(-3, -2)), s=patch.shape)
        #     ifft_patch2 = np.fft.irfftn(np.fft.ifftshift(fft_patch2, axes=(-3, -2)), s=patch.shape)
            
        #     return ifft_patch1.astype(np.float32), ifft_patch2.astype(np.float32)

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
