"""Shared test fixtures for the Easel test suite."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import pytest
from easel import Data, Model, Engine


class DummyData(Data):
    """A simple Data subclass with a 100x10 -> 100x1 regression dataset."""

    def setup(self, stage=None):
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        ds = TensorDataset(x, y)
        self.train_dataset = ds
        self.val_dataset = ds
        self.test_dataset = ds
        self.predict_dataset = ds


class LinearModel(Model):
    """A single-layer linear model for testing."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": 1, "strategy": "epoch"},
        }


@pytest.fixture
def dummy_data():
    """Return a fresh DummyData instance."""
    return DummyData()


@pytest.fixture
def linear_model():
    """Return a fresh LinearModel instance."""
    return LinearModel()


@pytest.fixture
def make_engine():
    """Return a factory function that builds Engine instances.

    Usage:
        engine = make_engine(max_epochs=5, train_batch_size=8)
    """

    def _make(**overrides):
        defaults = dict(
            model=LinearModel(),
            data=DummyData(),
            train_batch_size=16,
            seed=42,
            max_epochs=2,
        )
        defaults.update(overrides)
        return Engine(**defaults)

    return _make
