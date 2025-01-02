# train.py
import torch
import pytorch_lightning as L
import time


# STEP 1: DEFINE YOUR LIGHTNING MODULE
class ToyExample(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch):
        print("Rank", self.local_rank)
        # Send the batch through the model and calculate the loss
        # The Trainer will run .backward(), optimizer.step(), .zero_grad(), etc. for you
        loss = self.model(batch).sum()
        time.sleep(0.1)
        return loss

    def configure_optimizers(self):
        # Choose an optimizer or implement your own.
        return torch.optim.Adam(self.model.parameters())


# STEP 2: RUN THE TRAINER
if __name__ == "__main__":
    # Set up the model so it can be called in `training_step`.
    # This is a dummy model. Replace it with an LLM or whatever
    model = torch.nn.Linear(32, 2)
    pl_module = ToyExample(model)
    # Configure the dataset and return a data loader.
    train_dataloader = torch.utils.data.DataLoader(
        torch.randn(1000000, 32), num_workers=1
    )
    trainer = L.Trainer(strategy="ddp", accelerator="gpu", devices=[2, 3])
    trainer.fit(pl_module, train_dataloader)
