"""Tests for Engine._get_dataloader_kwargs and dataloader construction."""


def test_dataloader_kwargs_default_batch_size(make_engine):
    """Default batch_size should reach the underlying train DataLoader."""
    engine = make_engine(train_batch_size=8)

    assert engine.train_dataloader.batch_sampler.batch_size == 8


def test_dataloader_kwargs_global_applied(make_engine):
    """Global dataloader_config keys should reach the DataLoader."""
    engine = make_engine(dataloader_config={"shuffle": False})

    assert engine.train_dataloader.batch_sampler.sampler.__class__.__name__ == "SequentialSampler"


def test_dataloader_kwargs_mode_section_applied(make_engine):
    """Per-mode section should override the default batch_size."""
    engine = make_engine(dataloader_config={"train": {"batch_size": 8}})

    assert engine.train_dataloader.batch_sampler.batch_size == 8
