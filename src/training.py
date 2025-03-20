import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from utils.utils import set_seed
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf
import uuid


torch.set_float32_matmul_precision("medium")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)
    # to delete
    # from models.denoiseg import Denoiseg
    # model = Denoiseg.load_from_checkpoint(cfg.ckpt_path, config=cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method["train_batch_size"] // len(cfg.devices), 1)
    eval_batch_size = max(cfg.method["eval_batch_size"] // len(cfg.devices), 1)

    call_backs = []

    exp_name = f"{cfg.exp_name}-{str(uuid.uuid4())[:5]}"
    checkpoint_callback = ModelCheckpoint(
        monitor="val/dice_loss",  # Replace with your validation metric
        filename="{epoch}-{val/dice_loss:.2f}",
        save_top_k=5,
        mode="min",  # 'min' for loss/error, 'max' for accuracy
        dirpath=f"./ttt_ckpt/{exp_name}",
    )
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")

    call_backs.extend([checkpoint_callback, learning_rate_monitor])

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        num_workers=cfg.train_load_num_workers,
        shuffle=cfg.shuffle,
        drop_last=False,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        # collate_fn=train_set.collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=eval_batch_size,
        num_workers=cfg.val_load_num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        # collate_fn=train_set.collate_fn,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=(
            None
            if cfg.debug
            else WandbLogger(project=cfg.wandb_project_name, name=exp_name, id=exp_name)
        ),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler=cfg.profiler,
        # strategy="auto" if cfg.debug else "ddp",
        # strategy="auto",
        strategy=cfg.strategy,
        callbacks=call_backs,
        check_val_every_n_epoch=cfg.check_val_every_n_epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        enable_progress_bar=cfg.enable_progress_bar,
    )
    print(trainer.strategy)

    # automatically resume training
    # if cfg.ckpt_path is None and not cfg.debug:
    #     # Pattern to match all .ckpt files in the base_path recursively
    #     search_pattern = os.path.join("./src", exp_name, "**", "*.ckpt")
    #     cfg.ckpt_path = find_latest_checkpoint(search_pattern)
    #     print(f"Found checkpoint: {cfg.ckpt_path}")

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path,
    )


if __name__ == "__main__":
    train()
