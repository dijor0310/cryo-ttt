import h5py
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pickle
import torch


class DenoisegPatchDataset(Dataset):
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
            # print(list(config.keys()))
            self.root_dir = Path(config.test_data_root_dir)
            # print(self.root_dir)
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

        self.mask_ratio = config.mask_ratio
        self.window_size = config.window_size

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

        # masked_raw, mask = self.generate_mask(deepcopy(raw))
        masked_raw, mask = self.generate_mask(raw)
        # masked_raw, mask = raw, raw

        return {
            "image": masked_raw[np.newaxis, ...].astype(np.float32),
            "unmasked_image": raw[np.newaxis, ...].astype(np.float32),
            "mask": mask[np.newaxis, ...].astype(np.float32),
            "label": label[np.newaxis, ...].astype(np.int8),
            "id": hdf5_file.parents[0].stem + "_" + raw_key,
        }

    def generate_mask(self, input_image):
        """
        Generates a masked version of the input 3D image by replacing selected voxels with their neighboring voxel values.

        Parameters:
        - input_image (numpy.ndarray): The input noisy 3D image with shape (size, size, size).
        - mask_ratio (float): The proportion of voxels to be masked (0 < mask_ratio < 1).
        - window_size (tuple): The size of the local neighborhood around each voxel (depth, height, width).

        Returns:
        - output_image (numpy.ndarray): The modified image with masked voxels.
        - mask (numpy.ndarray): A binary mask indicating the locations of the masked voxels.
        """
        # Validate inputs
        # if not (0 < mask_ratio < 1):
        #     raise ValueError("mask_ratio must be between 0 and 1.")
        # if not (isinstance(window_size, tuple) and len(window_size) == 3):
        # raise ValueError("window_size must be a tuple of three integers.")

        # Get image dimensions
        depth, height, width = input_image.shape
        num_samples = int(depth * height * width * (self.mask_ratio))

        # Initialize mask and output image
        mask = np.zeros_like(input_image)
        output_image = np.copy(input_image)

        # # Define a grid of coordinates for each axis in the input patch and the step size
        # pixel_coords = []
        # steps = []
        # # mask_pixel_distance = (self.mask_ratio * depth * height * width) ** (1.0 / 3.0)
        # for axis_size in input_image.shape:
        #     # make sure axis size is evenly divisible by box size
        #     num_pixels = (self.mask_ratio ** (1.0 / 3.0)) * axis_size
        #     axis_pixel_coords, step = np.linspace(
        #         0, axis_size, num_pixels, dtype=np.int32, endpoint=False, retstep=True
        #     )
        #     # explain
        #     pixel_coords.append(axis_pixel_coords.T)
        #     steps.append(step)
        # coordinate_grid_list = np.meshgrid(*pixel_coords)
        # coordinate_grid = np.array(coordinate_grid_list).reshape(len(input_image.shape), -1).T
        # coordinate_grid += grid_random_increment
        # coordinate_grid = np.clip(coordinate_grid, 0, np.array(shape) - 1)

        # Generate random indices for masked voxels
        # mask_indices = np.zeros(num_samples)
        mask_indices = np.random.choice(
            depth * height * width, num_samples, replace=False
        )
        mask_d, mask_h, mask_w = np.unravel_index(mask_indices, (depth, height, width))

        # Generate random offsets for neighboring voxels
        offset_d = np.random.randint(
            -self.window_size // 2, self.window_size // 2 + 1, num_samples
        )
        offset_h = np.random.randint(
            -self.window_size // 2, self.window_size // 2 + 1, num_samples
        )
        offset_w = np.random.randint(
            -self.window_size // 2, self.window_size // 2 + 1, num_samples
        )

        # Calculate neighboring indices with boundary handling
        neighbor_d = (mask_d + offset_d) % depth
        neighbor_h = (mask_h + offset_h) % height
        neighbor_w = (mask_w + offset_w) % width

        # Replace masked voxels with their neighboring voxel values
        output_image[mask_d, mask_h, mask_w] = input_image[
            neighbor_d, neighbor_h, neighbor_w
        ]
        mask[mask_d, mask_h, mask_w] = 1  # Mark masked voxels

        return output_image, mask


class DenoisegPatchInMemoryDataset(Dataset):
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
            # print(list(config.keys()))
            self.root_dir = Path(config.test_data_root_dir)
            # print(self.root_dir)
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
                keys = list(f["volumes"]["raw"].keys())
                subtomos = [f[f"volumes/raw/{key}"][...] for key in keys]
                labels = [f[f"volumes/labels/memb/{key}"][...] for key in keys]

                self.data_keys.extend(
                    [
                        (file_idx, raw_key, label_key)
                        for raw_key, label_key in zip(subtomos, labels)
                    ]
                )

        self.mask_ratio = config.mask_ratio
        self.window_size = config.window_size

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        file_idx, raw, label = self.data_keys[idx]
        hdf5_file = self.hdf5_files[file_idx]

        # with h5py.File(hdf5_file, "r") as f:
        #     # Load raw subtomogram and label
        #     raw = f[f"volumes/raw/{raw_key}"][...]
        #     label = f[f"{label_key}"][...]
        # # Convert to tensors
        # raw = torch.tensor(raw, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.long)

        # Normalize raw subtomogram if required
        if self.normalize:
            raw = (raw - raw.mean()) / (raw.std() + 1e-8)

        # Apply transformations if provided
        if self.transform:
            raw = self.transform(raw)

        # masked_raw, mask = self.generate_mask(deepcopy(raw))
        masked_raw, mask = self.generate_mask(raw)
        # masked_raw, mask = raw, raw

        return {
            "image": masked_raw[np.newaxis, ...].astype(np.float32),
            "unmasked_image": raw[np.newaxis, ...].astype(np.float32),
            "mask": mask[np.newaxis, ...].astype(np.float32),
            "label": label[np.newaxis, ...].astype(np.int8),
            "id": hdf5_file.parents[0].stem + "_",
        }

    def generate_mask(self, input_image):
        """
        Generates a masked version of the input 3D image by replacing selected voxels with their neighboring voxel values.

        Parameters:
        - input_image (numpy.ndarray): The input noisy 3D image with shape (size, size, size).
        - mask_ratio (float): The proportion of voxels to be masked (0 < mask_ratio < 1).
        - window_size (tuple): The size of the local neighborhood around each voxel (depth, height, width).

        Returns:
        - output_image (numpy.ndarray): The modified image with masked voxels.
        - mask (numpy.ndarray): A binary mask indicating the locations of the masked voxels.
        """
        # Validate inputs
        # if not (0 < mask_ratio < 1):
        #     raise ValueError("mask_ratio must be between 0 and 1.")
        # if not (isinstance(window_size, tuple) and len(window_size) == 3):
        # raise ValueError("window_size must be a tuple of three integers.")

        # Get image dimensions
        depth, height, width = input_image.shape
        num_samples = int(depth * height * width * (self.mask_ratio))

        # Initialize mask and output image
        mask = np.zeros_like(input_image)
        output_image = np.copy(input_image)

        # # Define a grid of coordinates for each axis in the input patch and the step size
        # pixel_coords = []
        # steps = []
        # # mask_pixel_distance = (self.mask_ratio * depth * height * width) ** (1.0 / 3.0)
        # for axis_size in input_image.shape:
        #     # make sure axis size is evenly divisible by box size
        #     num_pixels = (self.mask_ratio ** (1.0 / 3.0)) * axis_size
        #     axis_pixel_coords, step = np.linspace(
        #         0, axis_size, num_pixels, dtype=np.int32, endpoint=False, retstep=True
        #     )
        #     # explain
        #     pixel_coords.append(axis_pixel_coords.T)
        #     steps.append(step)
        # coordinate_grid_list = np.meshgrid(*pixel_coords)
        # coordinate_grid = np.array(coordinate_grid_list).reshape(len(input_image.shape), -1).T
        # coordinate_grid += grid_random_increment
        # coordinate_grid = np.clip(coordinate_grid, 0, np.array(shape) - 1)

        # Generate random indices for masked voxels
        # mask_indices = np.zeros(num_samples)
        mask_indices = np.random.choice(
            depth * height * width, num_samples, replace=False
        )
        mask_d, mask_h, mask_w = np.unravel_index(mask_indices, (depth, height, width))

        # Generate random offsets for neighboring voxels
        offset_d = np.random.randint(
            -self.window_size // 2, self.window_size // 2 + 1, num_samples
        )
        offset_h = np.random.randint(
            -self.window_size // 2, self.window_size // 2 + 1, num_samples
        )
        offset_w = np.random.randint(
            -self.window_size // 2, self.window_size // 2 + 1, num_samples
        )

        # Calculate neighboring indices with boundary handling
        neighbor_d = (mask_d + offset_d) % depth
        neighbor_h = (mask_h + offset_h) % height
        neighbor_w = (mask_w + offset_w) % width

        # Replace masked voxels with their neighboring voxel values
        output_image[mask_d, mask_h, mask_w] = input_image[
            neighbor_d, neighbor_h, neighbor_w
        ]
        mask[mask_d, mask_h, mask_w] = 1  # Mark masked voxels

        return output_image, mask


class DenoisegPatchDatasetV2(Dataset):
    def __init__(self, config, is_validation, is_test):
        """
        Args:
            root_dir (string): Root directory containing tomogram subdirectories.
            tomo_names (list of strings): List of tomogram names to include.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        if is_validation:
            self.root_dir = Path(config.val_data_root_dir)
            self.tomo_names = config.val_tomo_names
        elif is_test:
            self.root_dir = Path(config.test_data_root_dir)
            self.tomo_names = config.test_tomo_names
        else:
            self.root_dir = Path(config.train_data_root_dir)
            self.tomo_names = config.train_tomo_names

        # root_path = Path(self.root_dir)
        self.file_paths = [
            file
            for tomo_name in self.tomo_names
            for file in self.root_dir.joinpath(tomo_name).rglob("*.pkl")
        ]

        self.mask_ratio = config.mask_ratio
        self.window_size = config.window_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with open(file_path, "rb") as f:
            sample = pickle.load(f)

        masked_raw, mask = self.generate_mask(sample["subtomo"])

        # print(sample["start_coord"])
        return {
            "image": masked_raw.unsqueeze(0).to(torch.float32),
            "unmasked_image": sample["subtomo"].unsqueeze(0).to(torch.float32),
            "mask": mask.unsqueeze(0).to(torch.uint8),
            "label": sample["label"].unsqueeze(0).to(torch.uint8),
            "id": sample["tomo_name"]
            + "/"
            + "_".join(str(start_coord) for start_coord in sample["start_coord"]),
            "start_coord": torch.Tensor(sample["start_coord"]).to(torch.int32),
        }

    def generate_mask(self, input_image):
        """
        Generates a masked version of the input 3D image by replacing selected voxels with their neighboring voxel values.

        Parameters:
        - input_image (numpy.ndarray): The input noisy 3D image with shape (size, size, size).
        - mask_ratio (float): The proportion of voxels to be masked (0 < mask_ratio < 1).
        - window_size (tuple): The size of the local neighborhood around each voxel (depth, height, width).

        Returns:
        - output_image (numpy.ndarray): The modified image with masked voxels.
        - mask (numpy.ndarray): A binary mask indicating the locations of the masked voxels.
        """

        # Get image dimensions
        depth, height, width = input_image.shape
        num_samples = int(depth * height * width * (self.mask_ratio))

        # Initialize mask and output image
        input_image = input_image.numpy()
        mask = np.zeros_like(input_image)
        output_image = np.copy(input_image)

        mask_indices = np.random.choice(
            depth * height * width, num_samples, replace=False
        )
        mask_d, mask_h, mask_w = np.unravel_index(mask_indices, (depth, height, width))

        # Generate random offsets for neighboring voxels
        offset_d = np.random.randint(
            -self.window_size // 2, self.window_size // 2 + 1, num_samples
        )
        offset_h = np.random.randint(
            -self.window_size // 2, self.window_size // 2 + 1, num_samples
        )
        offset_w = np.random.randint(
            -self.window_size // 2, self.window_size // 2 + 1, num_samples
        )

        # Calculate neighboring indices with boundary handling
        neighbor_d = (mask_d + offset_d) % depth
        neighbor_h = (mask_h + offset_h) % height
        neighbor_w = (mask_w + offset_w) % width

        # Replace masked voxels with their neighboring voxel values
        output_image[mask_d, mask_h, mask_w] = input_image[
            neighbor_d, neighbor_h, neighbor_w
        ]
        mask[mask_d, mask_h, mask_w] = 1  # Mark masked voxels

        return torch.from_numpy(output_image), torch.from_numpy(mask)
