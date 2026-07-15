"""Tests for the Engine.run() orchestrator."""

import torch.nn as nn
from easel import Engine
from conftest import DummyData, LinearModel


def test_val_always_runs_after_training():
    """run() should call run_val after run_train, even if val_strategy='no'."""
    val_count = 0

    class TrainValEngine(Engine):
        def train_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return nn.functional.mse_loss(preds, y)

        def val_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return {"val_loss": nn.functional.mse_loss(preds, y)}

        def on_val_start(self):
            nonlocal val_count
            val_count += 1

    engine = TrainValEngine(
        model=LinearModel(),
        data=DummyData(),
        train_batch_size=32,
        seed=42,
        max_epochs=2,
        val_strategy="no",
        do_test=False,
        do_predict=False,
    )
    engine.run()

    assert val_count >= 1, "Val should run after training even with val_strategy='no'"
