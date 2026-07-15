"""Tests for Engine.setup_model (inference path)."""

from easel import Model, Engine
from conftest import DummyData


def test_inference_mode_no_optimizers():
    """do_train=False should skip optimizer setup entirely."""
    class NoOptModel(Model):
        def forward(self, x):
            return x
        def configure_optimizers(self):
            return None

    engine = Engine(
        model=NoOptModel(),
        data=DummyData(),
        do_train=False,
        do_test=True,
        eval_batch_size=4,
    )

    assert len(engine.optimizers) == 0
    assert len(engine.schedulers) == 0
