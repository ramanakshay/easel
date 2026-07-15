"""Tests for Engine.setup_data."""

import math
import pytest
import torch
from torch.utils.data import TensorDataset, IterableDataset

from easel import Data, Engine
from conftest import DummyData, LinearModel


def test_dataloaders_created(make_engine):
    """Engine should build train and val dataloaders by default."""
    engine = make_engine()

    assert engine.train_dataloader is not None
    assert engine.val_dataloader is not None


def test_train_steps_per_epoch_auto_calculated(make_engine):
    """train_steps_per_epoch should be derived from dataset size and batch size."""
    engine = make_engine(train_batch_size=32)

    assert engine.train_steps_per_epoch == math.ceil(100 / 32)


def test_train_steps_per_epoch_with_grad_accum(make_engine):
    """train_steps_per_epoch should be divided by gradient_accumulation_steps."""
    engine = make_engine(train_batch_size=32, gradient_accumulation_steps=4)

    expected = math.ceil(math.ceil(100 / 32) / 4)
    assert engine.train_steps_per_epoch == expected


def test_max_steps_derived_from_max_epochs(make_engine):
    """max_steps = max_epochs * train_steps_per_epoch when max_steps is not set."""
    engine = make_engine(max_epochs=3, train_steps_per_epoch=10)

    assert engine.max_steps == 30


def test_max_epochs_derived_from_max_steps(make_engine):
    """max_epochs = ceil(max_steps / train_steps_per_epoch) when max_epochs is not set."""
    engine = make_engine(max_epochs=None, max_steps=50, train_steps_per_epoch=20)

    assert engine.max_epochs == math.ceil(50 / 20)


def test_both_max_epochs_and_max_steps_none_raises(make_engine):
    """do_train=True requires at least one of max_epochs or max_steps."""
    with pytest.raises(ValueError, match="At least one"):
        make_engine(max_epochs=None, max_steps=None)


def test_iterable_dataset_max_epochs_only_raises():
    """IterableDataset with max_epochs but no train_steps_per_epoch should raise."""
    class IterableData(Data):
        def setup(self, stage=None):
            class IterDS(IterableDataset):
                def __iter__(self):
                    while True:
                        yield torch.randn(10), torch.randn(1)
            self.train_dataset = IterDS()

    with pytest.raises(ValueError, match="IterableDataset"):
        Engine(
            model=LinearModel(),
            data=IterableData(),
            train_batch_size=4,
            max_epochs=1,
            seed=42,
        )


def test_modes_disabled_when_loader_none():
    """Modes with no dataset should be auto-disabled."""
    class PartialData(Data):
        def setup(self, stage=None):
            self.train_dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 1))

    engine = Engine(
        model=LinearModel(),
        data=PartialData(),
        train_batch_size=4,
        max_epochs=1,
        seed=42,
    )

    assert engine.do_train is True
    assert engine.do_val is False
    assert engine.do_test is False
    assert engine.do_predict is False
