"""Tests for standalone run_val, run_test, run_predict (no training)."""

import torch
import torch.nn as nn
from easel import Engine
from conftest import DummyData, LinearModel


def test_standalone_val():
    """run_val should process all batches from the val dataloader."""
    batch_count = 0

    class ValEngine(Engine):
        def val_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return {"val_loss": nn.functional.mse_loss(preds, y)}

        def on_val_step_end(self, outputs, batch, batch_idx):
            nonlocal batch_count
            batch_count += 1

    engine = ValEngine(
        model=LinearModel(),
        data=DummyData(),
        do_train=False,
        do_val=True,
        eval_batch_size=4,
    )
    engine.run_val()

    # DummyData has 100 samples / batch_size 4 = 25 batches.
    assert batch_count == 25


def test_standalone_test():
    """run_test should process all batches from the test dataloader."""
    batch_count = 0

    class TestEngine(Engine):
        def test_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return {"test_loss": nn.functional.mse_loss(preds, y)}

        def on_test_step_end(self, outputs, batch, batch_idx):
            nonlocal batch_count
            batch_count += 1

    engine = TestEngine(
        model=LinearModel(),
        data=DummyData(),
        do_train=False,
        do_test=True,
        eval_batch_size=4,
    )
    engine.run_test()

    assert batch_count == 25


def test_standalone_predict():
    """run_predict should process all batches from the predict dataloader."""
    batch_count = 0

    class PredictEngine(Engine):
        def predict_step(self, batch):
            x, y = batch
            return self.model(x)

        def on_predict_step_end(self, outputs, batch, batch_idx):
            nonlocal batch_count
            batch_count += 1

    engine = PredictEngine(
        model=LinearModel(),
        data=DummyData(),
        do_train=False,
        do_predict=True,
        eval_batch_size=4,
    )
    engine.run_predict()

    assert batch_count == 25
