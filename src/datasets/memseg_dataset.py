import os
import torch
import nibabel as nib
from torch.utils.data import Dataset
from typing import Dict
import numpy as np
from pathlib import Path
import pickle

class MemSegDataset(Dataset):
    def __init__(self, config, is_validation=False, is_test=False):

        self.is_test = is_test
        self.is_validation = is_validation

        if is_test or is_validation:
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
        else:
            self.image_dir = os.path.join(config.root_dir, "imagesTr")
            self.label_dir = os.path.join(config.root_dir, "labelsTr")

        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".nii.gz")])
        self.label_filenames = [f.rsplit("_", 1)[0] + ".nii.gz" for f in self.image_filenames]
        
    def __len__(self):
        return len(self.image_filenames)
    
    def load_nii(self, filepath):
        nii_img = nib.load(filepath)
        return np.array(nii_img.get_fdata(), dtype=np.float32)
    
    def __getitem__(self, idx):
        
        if self.is_test or self.is_validation:
            return self.get_test_sample(idx)

        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])
        
        image = self.load_nii(img_path)
        label = self.load_nii(label_path)

        if self.config.normalize_data:
            image = (image - image.mean()) / (image.std() + 1e-8)
                
        sample = {
            "image": image[np.newaxis, ...],
            "label": label[np.newaxis, ...]
        }

        sample = self.get_random_crop(sample)
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
        image = image.swapaxes(-1, -3)
        label = label.swapaxes(-1, -3)
        start_coord[..., [0, 2]] = start_coord[..., [2, 0]]

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
                constant_values=2,
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
                constant_values=0,
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
                    constant_values=2,
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
                    constant_values=0,
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

# Example usage:
# dataset = NiiDataset("/path/to/data")
# img, label = dataset[0]
