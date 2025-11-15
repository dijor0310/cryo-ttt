import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from models import build_model
from datasets import build_dataset
from ttt.utils.utils import set_seed
import hydra
from omegaconf import OmegaConf
import uuid

# We can investigate this at some time, for now should be OK
torch.set_float32_matmul_precision("medium")  # for A* series GPUs


@hydra.main(version_base=None, config_path="configs", config_name="testing")
def test(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    test_set = build_dataset(cfg, test=True)

    test_batch_size = max(cfg.method["test_batch_size"] // len(cfg.devices), 1)

    call_backs = []

    exp_name = f"{cfg.exp_name}-{str(uuid.uuid4())[:5]}"
    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val/dice_loss",  # Replace with your validation metric
    #     filename="{epoch}-{val/dice_loss:.2f}",
    #     save_top_k=5,
    #     mode="min",  # 'min' for loss/error, 'max' for accuracy
    #     dirpath=f"./ttt_ckpt/{exp_name}",
    # )
    # learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")

    # call_backs.extend([checkpoint_callback, learning_rate_monitor])

    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        num_workers=cfg.test_load_num_workers,
        shuffle=cfg.shuffle,
        drop_last=False,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=(
            None
            if cfg.debug
            else WandbLogger(
                project=cfg.wandb_project_name,
                name=exp_name,
                id=exp_name,
                config=OmegaConf.to_container(cfg),
            )
        ),
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler=cfg.profiler,
        strategy=cfg.strategy,
        callbacks=call_backs,
        check_val_every_n_epoch=cfg.check_val_every_n_epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        enable_progress_bar=cfg.enable_progress_bar,
    )

    trainer.test(
        model=model,
        dataloaders=test_loader,
        ckpt_path=cfg.ckpt_path,
    )


if __name__ == "__main__":
    test()
