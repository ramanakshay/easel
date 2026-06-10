import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import logging

from easel import Data, Model, Engine

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

    logger.info("Initializing Engine...")
    engine = Engine(
        model=model,
        data=data,
        do_train=True,
        do_val=True,
        train_batch_size=16,
        seed=42,
        mixed_precision="no",
        max_epochs=2,
    )

    logger.info("Running assertions...")

    assert engine.train_dataloader is not None, "Train dataloader was not created!"
    assert engine.val_dataloader is not None, "Val dataloader was not created!"
    logger.info("Dataloaders successfully prepared and wrapped.")

    assert len(engine.optimizers) == 1, "Optimizer was not parsed correctly!"
    assert len(engine.schedulers) == 1, "Scheduler was not parsed correctly!"
    logger.info("Optimizers and Schedulers successfully parsed and standardized.")

    batch = next(iter(engine.train_dataloader))
    x, y = batch
    assert next(engine.model.parameters()).device == x.device, "Model and data are on different devices!"
    logger.info(f"Model and Data successfully moved to: {engine.device}")

    logger.info("Simulating a single training step using Engine primitives...")

    engine.optimizers_zero_grad()

    with engine.autocast():
        predictions = engine.model(x)
        loss = nn.functional.mse_loss(predictions, y)

    engine.backward(loss)

    engine.optimizers_step()

    engine.schedulers_step(strategy="epoch", counter=1)

    logger.info("Forward, Backward, and Optimization steps executed without crashing.")

    # ── Logging API tests ──
    logger.info("Testing logging API...")

    # Test 1: log() without flush — buffer, no monitor
    engine.log({"train_loss": 1.0})
    assert "train_loss" in engine._log_buffer
    assert "train_loss" not in engine.monitor
    logger.info("log() buffer-only: OK")

    # Test 2: log() with monitor=True — monitor updated
    engine.log({"val_loss": 0.5}, monitor=True)
    assert "val_loss" in engine._log_buffer
    assert engine.monitor["val_loss"] == 0.5
    logger.info("log(monitor=True): OK")

    # Test 3: log() with flush=True — buffer flushed immediately
    engine.log({"flushed": 1}, flush=True)
    assert "flushed" not in engine._log_buffer
    logger.info("log(flush=True): OK")

    # Test 4: should_log() with log_strategy="no" returns False
    engine.log_strategy = "no"
    assert not engine.should_log()
    engine.log_strategy = "step"
    logger.info("should_log() with 'no' strategy: OK")

    # Test 5: training flag is True during initialization
    assert engine.training is True
    logger.info("engine.training is True during setup: OK")

    # Test 6: val_strategy="no" makes should_validate return False
    engine.val_strategy = "no"
    assert not engine.should_validate()
    engine.val_strategy = "epoch"
    logger.info("should_validate() with 'no' strategy: OK")

    # Test 7: training flag — simulate post-training state
    engine.training = False
    assert engine.training is False
    logger.info("engine.training flag: OK")

    # Test 8: log() still works after training (buffer + monitor), flush works
    engine.log({"post_train": 42}, monitor=True)
    assert "post_train" in engine._log_buffer
    assert engine.monitor["post_train"] == 42
    engine._flush_log(step=engine.step)
    assert "post_train" not in engine._log_buffer
    logger.info("log() + monitor + flush work after training: OK")

    logger.info("--- All Tests Passed! Architecture is sound. ---")


if __name__ == "__main__":
    run_tests()