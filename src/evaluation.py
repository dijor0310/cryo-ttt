import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
import hydra
from omegaconf import OmegaConf
import uuid


torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluate(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    test_set = build_dataset(cfg, test=True)

    test_batch_size = max(cfg.method["test_batch_size"] // len(cfg.devices), 1)

    exp_name = f"{cfg.exp_name}-{str(uuid.uuid4())[:5]}"

    print(f"Dataset length: {len(test_set)}")
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        num_workers=cfg.load_num_workers,
        shuffle=False,
        drop_last=False,
        persistent_workers=cfg.persistent_workers,
        pin_memory=cfg.pin_memory,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=(
            None
            if cfg.debug
            else WandbLogger(project="cryo-ttt-testing", name=exp_name, id=exp_name)
        ),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        # strategy="auto" if cfg.debug else "ddp",
        strategy="auto",
        log_every_n_steps=cfg.log_every_n_steps,
        benchmark=cfg.benchmark,
    )

    trainer.test(
        model=model,
        ckpt_path=cfg.ckpt_path,
        dataloaders=test_loader,
        verbose=True,
    )


if __name__ == "__main__":
    evaluate()
