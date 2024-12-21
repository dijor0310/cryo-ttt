# Set torch precision
# torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf
from datasets.tomo_dataset import TomoDataset

# import os
from tqdm import tqdm


@hydra.main(version_base=None, config_path="configs", config_name="config")
def data_prep(cfg):
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    raw_dataset = TomoDataset(cfg)
    raw_dataloader = DataLoader(
        dataset=raw_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    for batch in tqdm(raw_dataloader):
        tomo, gt = batch

    return


if __name__ == "__main__":
    data_prep()
