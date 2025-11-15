# Set torch precision
# torch.set_float32_matmul_precision('medium')
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf
from ttt.datasets.tomo_dataset import TomoDataset
from ttt.utils.ddw_subtomos import extract_subtomos
import pickle
import torch
from pathlib import Path

# import os
from tqdm import tqdm


def crop_to_shape(tensor, shape):
    if len(shape) != 3 or tensor.ndim != 3:
        raise ValueError("Should be 3 dimensions!")

    return tensor[: shape[0], : shape[1], : shape[2]]


def normalize(tomo):
    return (tomo - tomo.mean()) / tomo.std()


@hydra.main(version_base=None, config_path="configs", config_name="data_prep")
def data_prep(cfg):
    OmegaConf.set_struct(cfg, False)  # Open the struct
    # cfg = OmegaConf.merge(cfg, cfg.method)

    raw_dataset = TomoDataset(cfg)
    raw_dataloader = DataLoader(
        dataset=raw_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for batch in tqdm(raw_dataloader):
        tomo, gt, tomo_name = batch

        tomo = tomo.squeeze()
        gt = gt.squeeze()
        tomo_name = tomo_name[0]

        tomo = crop_to_shape(tomo, gt.shape)

        if cfg.normalize:
            tomo = normalize(tomo)

        print(tomo.shape, gt.shape, gt.dtype)
        subtomos, subtomos_start_coord = extract_subtomos(
            tomo=tomo.to(torch.float32),
            subtomo_size=cfg.patch_size,
            subtomo_extraction_strides=[cfg.overlap, cfg.overlap, cfg.overlap],
            pad_before_subtomo_extraction=True,
        )
        subtomos_gt, subtomos_start_coord_sanity = extract_subtomos(
            tomo=gt.to(torch.float32),
            subtomo_size=cfg.patch_size,
            subtomo_extraction_strides=[cfg.overlap, cfg.overlap, cfg.overlap],
            pad_before_subtomo_extraction=True,
        )

        assert torch.allclose(
            torch.Tensor(subtomos_start_coord),
            torch.Tensor(subtomos_start_coord_sanity),
        ), "Start coordinates should be the same!"

        for subtomo, subtomo_gt, subtomo_start_coord in zip(
            subtomos, subtomos_gt, subtomos_start_coord
        ):
            sample = {
                "subtomo": subtomo.clone().to(torch.float32),
                "label": subtomo_gt.clone().to(torch.uint8),
                "tomo_name": tomo_name,
                "start_coord": subtomo_start_coord,
            }
            subtomo_name = "_".join(
                str(start_coord) for start_coord in subtomo_start_coord
            )

            Path(f"{cfg.dataset_path}/{tomo_name}").mkdir(exist_ok=True, parents=True)

            with open(f"{cfg.dataset_path}/{tomo_name}/{subtomo_name}.pkl", "wb") as f:
                pickle.dump(sample, f, protocol=5)
    return


if __name__ == "__main__":
    data_prep()
