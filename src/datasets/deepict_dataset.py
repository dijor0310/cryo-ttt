import h5py
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

# from torchvision.transforms import Normalize


class DeepictPatchDataset(Dataset):
    def __init__(self, config, is_validation, is_test):
        """
        PyTorch Dataset for loading subtomograms and labels from HDF5 files.

        Args:
            root_dir (str): Root directory to search for HDF5 files recursively.
            transform (callable, optional): Transform to apply to raw subtomograms.
            normalize (bool): Whether to normalize raw subtomograms.
        """
        if is_validation:
            self.root_dir = Path(config.val_data_root_dir)
        elif is_test:
            print(list(config.keys()))
            self.root_dir = Path(config.test_data_root_dir)
        else:
            self.root_dir = Path(config.train_data_root_dir)

        self.transform = None
        self.normalize = config.normalize_data

        # Find all HDF5 files recursively in the root directory
        self.hdf5_files = list(self.root_dir.rglob("*.h5"))

        # Store all keys to access data efficiently
        self.data_keys = []  # (file_index, raw_key, label_key)

        for file_idx, hdf5_file in enumerate(self.hdf5_files):
            with h5py.File(hdf5_file, "r") as f:
                raw_keys = list(f["volumes"]["raw"].keys())
                label_keys = [f"volumes/labels/memb/{key}" for key in raw_keys]

                self.data_keys.extend(
                    [
                        (file_idx, raw_key, label_key)
                        for raw_key, label_key in zip(raw_keys, label_keys)
                    ]
                )

        self.rotation_classes = [
            (None, 0),  # Identity
            (0, 90),
            (0, 180),
            (0, 270),  # rotation around x
            (1, 90),
            (1, 180),
            (1, 270),  # rotation around y
            (2, 90),
            (2, 180),
            (2, 270),  # rotation around z
        ]

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        file_idx, raw_key, label_key = self.data_keys[idx]
        hdf5_file = self.hdf5_files[file_idx]

        with h5py.File(hdf5_file, "r") as f:
            # Load raw subtomogram and label
            raw = f[f"volumes/raw/{raw_key}"][...]
            label = f[f"{label_key}"][...]

        # # Convert to tensors
        # raw = torch.tensor(raw, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.long)

        # Normalize raw subtomogram if required
        if self.normalize:
            raw = (raw - raw.mean()) / (raw.std() + 1e-8)

        # Apply transformations if provided
        if self.transform:
            raw = self.transform(raw)

        # rotation_idx = np.random.choice(len(self.rotation_classes))
        # axis, angle = self.rotation_classes[rotation_idx]

        # if axis is not None:
        #     # Apply the rotation
        #     rotated_raw = np.rot90(raw, k=angle // 90, axes=(axis, (axis + 1) % 3))
        # else:
        #     rotated_raw = raw  # Identity rotation

        # print(raw.shape, label.shape)
        return {
            "image": raw[np.newaxis, ...].astype(np.float32),
            "label": label[np.newaxis, ...].astype(np.int8),
            "id": hdf5_file.parents[0].stem + "_" + raw_key,
            # "rotated_image": rotated_raw[np.newaxis, ...].copy().astype(np.float32),
            # "rotation": rotation_idx,
        }
