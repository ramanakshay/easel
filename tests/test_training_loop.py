"""Tests for the training loop: run_train, max_steps, schedulers, should_stop."""

import torch
import torch.nn as nn
from easel import Engine
from conftest import DummyData, LinearModel


def _make_train_engine(model=None, data=None, **overrides):
    """Build an Engine subclass with train_step + val_step wired up."""
    class TrainEngine(Engine):
        def train_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return nn.functional.mse_loss(preds, y)

        def val_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return {"val_loss": nn.functional.mse_loss(preds, y)}

    defaults = dict(
        model=model or LinearModel(),
        data=data or DummyData(),
        train_batch_size=32,
        seed=42,
        max_steps=6,
    )
    defaults.update(overrides)
    return TrainEngine(**defaults)


def test_full_training_loop():
    engine = _make_train_engine(max_steps=6)

    assert engine.epoch == 0
    assert engine.step == 0

    engine.run_train()

    assert engine.step == 6


def test_train_step_returns_dict_with_loss():
    seen_outputs = []

    class DictTrainEngine(Engine):
        def train_step(self, batch):
            x, y = batch
            preds = self.model(x)
            loss = nn.functional.mse_loss(preds, y)
            return {"loss": loss, "extra": "metadata"}

        def on_train_substep_end(self, outputs, batch, batch_idx):
            seen_outputs.append(outputs)

    engine = DictTrainEngine(
        model=LinearModel(),
        data=DummyData(),
        train_batch_size=32,
        seed=42,
        max_steps=3,
    )
    engine.run_train()

    assert engine.step == 3
    assert all(isinstance(o, dict) and "loss" in o and o["extra"] == "metadata" for o in seen_outputs)


def test_training_loop_stops_at_max_steps():
    engine = _make_train_engine(max_steps=5)

    engine.run_train()

    assert engine.step == 5


def test_loop_updates_lr_scheduler():
    engine = _make_train_engine(max_epochs=2)

    lr_before = engine.optimizers[0].param_groups[0]["lr"]
    engine.run_train()
    lr_after = engine.optimizers[0].param_groups[0]["lr"]

    assert lr_after != lr_before, f"LR did not change: {lr_before} -> {lr_after}"


def test_should_stop_breaks_loop():
    class StopEngine(Engine):
        def train_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return nn.functional.mse_loss(preds, y)

        def on_train_step_end(self):
            if self.step >= 3:
                self.should_stop = True

    engine = StopEngine(
        model=LinearModel(),
        data=DummyData(),
        train_batch_size=32,
        seed=42,
        max_steps=10,
    )
    engine.run_train()

    assert engine.step == 3
