import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

# Import from your library
from easel import Module, DataModule, Trainer

# 1. Define a dummy dataset
class SimpleData(DataModule):
    def __init__(self):
        super().__init__()
        # Create 100 random samples (Input: 10 dim, Output: 1 dim)
        self.x = torch.randn(100, 10)
        self.y = torch.randn(100, 1)
        self.dataset = TensorDataset(self.x, self.y)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=10, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=10)

# 2. Define the training logic by subclassing Trainer
class SimpleTrainer(Trainer):
    def train_step(self, batch, batch_idx):
        x, y = batch
        # self.model is the wrapped Module; we call it directly
        preds = self.model(x)
        loss = nn.MSELoss()(preds, y)
        return loss

    def val_step(self, batch, batch_idx):
        x, y = batch
        preds = self.model(x)
        loss = nn.MSELoss()(preds, y)
        # Log validation loss to the progress bar/logger
        self.log({"val_loss": loss})
        return loss

# 3. Setup the components
def main():
    # A simple PyTorch model
    pytorch_model = nn.Linear(10, 1)
    optimizer = Adam(pytorch_model.parameters(), lr=0.01)

    # Wrap in Easel Module
    model = Module(model=pytorch_model, optimizers=optimizer)

    # Initialize Data
    data = SimpleData()

    # Initialize Trainer
    trainer = SimpleTrainer(
        model=model,
        data=data,
        max_epochs=3,
        experiment_name="test_run",
        accelerator_config={"cpu": True} # Force CPU for simple testing
    )

    # 4. Run!
    print("Starting training...")
    trainer.run()
    print("Training complete!")

if __name__ == "__main__":
    main()
