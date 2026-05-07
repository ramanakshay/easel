import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import logging

# Import your library components (adjust the import paths as needed)
from easel import Data, Model, Trainer
from easel.trainer.base import BaseTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DummyData(Data):
    def setup(self, stage=None):
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(x, y)

        self.train_dataset = dataset
        self.val_dataset = dataset


class DummyModel(Model):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return {
            "optimizer": optimizer,
            "scheduler": {
                "scheduler": scheduler,
                "interval": 1,
                "strategy": "epoch"
            }
        }


def run_tests():
    logger.info("--- Starting Architecture Tests ---")

    data = DummyData()
    model = DummyModel()

    logger.info("Initializing BaseTrainer...")
    trainer = BaseTrainer(
        model=model,
        data=data,
        do_train=True,
        do_val=True,
        train_batch_size=16,
        seed=42,
        mixed_precision="no"
    )

    logger.info("Running assertions...")

    assert trainer.train_dataloader is not None, "Train dataloader was not created!"
    assert trainer.val_dataloader is not None, "Val dataloader was not created!"
    logger.info("Dataloaders successfully prepared and wrapped.")

    assert len(trainer.optimizers) == 1, "Optimizer was not parsed correctly!"
    assert len(trainer.schedulers) == 1, "Scheduler was not parsed correctly!"
    logger.info("Optimizers and Schedulers successfully parsed and standardized.")

    batch = next(iter(trainer.train_dataloader))
    x, y = batch
    assert next(trainer.model.parameters()).device == x.device, "Model and data are on different devices!"
    logger.info(f"Model and Data successfully moved to: {trainer.device}")

    logger.info("Simulating a single training step using BaseTrainer primitives...")

    trainer.optimizers_zero_grad()

    with trainer.autocast():
        predictions = trainer.model(x)
        loss = nn.functional.mse_loss(predictions, y)

    trainer.backward(loss)

    trainer.optimizers_step()

    trainer.schedulers_step(strategy="epoch", counter=1)

    logger.info("Forward, Backward, and Optimization steps executed without crashing.")

    logger.info("Testing state saving...")
    trainer.save_state("./test_checkpoint")
    logger.info("State successfully saved.")

    logger.info("--- All Tests Passed! Architecture is sound. ---")


if __name__ == "__main__":
    run_tests()
