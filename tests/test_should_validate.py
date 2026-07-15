"""Tests for Engine.should_validate."""

import pytest


@pytest.mark.parametrize(
    "val_strategy,step,epoch,expected",
    [
        ("no",   5, 1, False),
        ("epoch", 0, 1, True),
        ("step",  5, 0, True),
    ],
    ids=["no_strategy", "epoch_strategy", "step_strategy"],
)
def test_should_validate(make_engine, val_strategy, step, epoch, expected):
    engine = make_engine()
    engine.val_strategy = val_strategy
    engine.step = step
    engine.epoch = epoch

    assert engine.should_validate() is expected


@pytest.mark.parametrize(
    "epoch,expected",
    [(2, False), (3, True)],
    ids=["below_start", "at_start"],
)
def test_should_validate_start(make_engine, epoch, expected):
    engine = make_engine(val_start=3, val_interval=1)
    engine.epoch = epoch

    assert engine.should_validate() is expected


@pytest.mark.parametrize(
    "epoch,expected",
    [(2, False), (3, True)],
    ids=["not_multiple", "is_multiple"],
)
def test_should_validate_interval(make_engine, epoch, expected):
    engine = make_engine(val_start=0, val_interval=3)
    engine.epoch = epoch

    assert engine.should_validate() is expected
