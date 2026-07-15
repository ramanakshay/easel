"""Tests for Engine.scheduler_step and schedulers_step."""

import torch
import torch.nn as nn
import pytest
from easel import Model, Engine
from conftest import DummyData


class _PlateauModel(Model):
    """Model with a ReduceLROnPlateau scheduler (requires a monitor)."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}


def test_scheduler_step_no_monitor(make_engine):
    """scheduler_step should work when monitor is None."""
    engine = make_engine()
    engine.schedulers[0]["monitor"] = None

    engine.optimizer_step(0)
    engine.scheduler_step(0)


def test_scheduler_step_with_monitor():
    """scheduler_step should read monitor value from engine.monitor and
    pass it to a ReduceLROnPlateau scheduler."""
    engine = Engine(
        model=_PlateauModel(),
        data=DummyData(),
        train_batch_size=4,
        max_epochs=1,
        seed=42,
    )
    engine.monitor["val_loss"] = 0.5

    engine.optimizer_step(0)
    engine.scheduler_step(0)


def test_scheduler_step_missing_monitor_raises(make_engine):
    """scheduler_step should raise KeyError if monitor value is missing."""
    engine = make_engine()
    engine.schedulers[0]["monitor"] = "val_loss"

    with pytest.raises(KeyError, match="val_loss"):
        engine.scheduler_step(0)


def test_schedulers_step_strategy_filtering(make_engine):
    """schedulers_step(strategy='epoch') should only step schedulers
    whose strategy is 'epoch', not 'step'."""
    engine = make_engine()

    # Record the LR before stepping.
    sched = engine.schedulers[0]["scheduler"]
    lr_before = sched.get_last_lr()[0]

    # Set the scheduler to 'step' strategy — it should NOT be stepped.
    engine.schedulers[0]["strategy"] = "step"
    engine.epoch = 1
    engine.schedulers_step(strategy="epoch")

    lr_after = sched.get_last_lr()[0]
    assert lr_after == lr_before, (
        f"Scheduler with strategy='step' was stepped during epoch strategy: "
        f"{lr_before} -> {lr_after}"
    )
