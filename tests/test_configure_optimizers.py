"""Tests for Engine._standardize_optimizers across all return formats."""

import pytest
import torch
import torch.nn as nn
from easel import Model, Engine
from conftest import DummyData


class _Base(Model):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
    def forward(self, x):
        return self.layer(x)


class _MultiLayer(Model):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 5)
        self.l2 = nn.Linear(5, 1)
    def forward(self, x):
        return self.l2(self.l1(x))


def _make_engine(model_class, **overrides):
    defaults = dict(
        model=model_class(),
        data=DummyData(),
        train_batch_size=4,
        max_epochs=1,
        seed=42,
    )
    defaults.update(overrides)
    return Engine(**defaults)


def test_single_optimizer():
    class M(_Base):
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    engine = _make_engine(M)
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 0


def test_tuple_opt_sched():
    class M(_Base):
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return opt, sched

    engine = _make_engine(M)
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 1


def test_tuple_opt_sched_dict():
    class M(_Base):
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return (opt, {"scheduler": sched, "interval": 1, "strategy": "epoch"})

    engine = _make_engine(M)
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 1


@pytest.mark.parametrize("sched_key", ["lr_scheduler", "scheduler"])
def test_dict_format(sched_key):
    class M(_Base):
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return {"optimizer": opt, sched_key: sched}

    engine = _make_engine(M)
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 1


def test_two_list_format():
    class M(_MultiLayer):
        def configure_optimizers(self):
            opt1 = torch.optim.SGD(self.l1.parameters(), lr=0.01)
            opt2 = torch.optim.SGD(self.l2.parameters(), lr=0.01)
            sched1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=1)
            return [opt1, opt2], [sched1]

    engine = _make_engine(M)
    assert len(engine.optimizers) == 2
    assert len(engine.schedulers) == 1


def test_two_optimizers_no_scheduler():
    class M(_MultiLayer):
        def configure_optimizers(self):
            opt1 = torch.optim.SGD(self.l1.parameters(), lr=0.01)
            opt2 = torch.optim.SGD(self.l2.parameters(), lr=0.01)
            return [opt1, opt2]

    engine = _make_engine(M)
    assert len(engine.optimizers) == 2
    assert len(engine.schedulers) == 0


def test_list_of_dicts():
    class M(_Base):
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return [{"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": 1, "strategy": "epoch"}}]

    engine = _make_engine(M)
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 1


def test_reduce_on_plateau_no_monitor_raises():
    class M(_Base):
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
            return {"optimizer": opt, "lr_scheduler": sched}

    with pytest.raises(ValueError, match="ReduceLROnPlateau"):
        _make_engine(M)


def test_none_return_with_do_train_raises():
    class M(_Base):
        def configure_optimizers(self):
            return None

    with pytest.raises(ValueError, match="at least one optimizer"):
        _make_engine(M)
