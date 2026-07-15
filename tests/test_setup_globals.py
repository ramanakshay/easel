"""Tests for Engine.setup_globals."""

import os
import torch
import pytest


def test_deterministic_sets_env_and_disables_benchmark(make_engine):
    """deterministic=True should set CUBLAS_WORKSPACE_CONFIG and disable
    cudnn_benchmark (even if the user requested it)."""
    engine = make_engine(deterministic=True, cudnn_benchmark=True)

    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert engine.cudnn_benchmark is False
    assert torch.backends.cudnn.deterministic is True


@pytest.mark.parametrize("tf32_value", [True, "medium"])
def test_tf32_enables_matmul_precision(make_engine, tf32_value):
    """tf32=True or tf32='medium' should enable TF32 on CUDA matmul and cudnn."""
    make_engine(tf32=tf32_value)

    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
