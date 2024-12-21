import torch
from torch.utils.data import Dataset
import mrcfile
import pandas as pd


class TomoDataset(Dataset):
    def __init__(self, config):
        self.config = config
        tomo_list = self.config.tomo_list
        self.tomo_paths = []
        self.gt_paths = []
        self.tomo_names = []

        for tomo_name, paths in tomo_list.items():
            self.tomo_paths.append(paths["tomo"])
            self.gt_paths.append(paths["membrane"])
            self.tomo_names.append(tomo_name)


    def __len__(self):
        return len(self.tomo_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        tomogram = mrcfile.read(self.tomo_paths[idx])
        segmentation_gt = mrcfile.read(self.gt_paths[idx])
        tomo_name = self.tomo_names[idx]

        # if self.transform:
        #     tomogram = self.transform(tomogram)
        #     segmentation_gt = self.transform(segmentation_gt)

        return tomogram, segmentation_gt, tomo_name
