"""Easel: a modular, generalizable deep learning library for SL tasks.

Easel provides a lightweight training and inference framework built on top
of PyTorch and HuggingFace ``accelerate``. Users subclass :class:`Model` and
:class:`Data`, implement a handful of ``*_step`` methods on an
:class:`Engine` subclass, and call :meth:`Engine.run`.

Public objects:
    Engine: Orchestrates the training, validation, testing, and prediction
        loops.
    Model: Base class for user-defined network architectures and optimizer
        configuration.
    Data: Base class for preparing datasets and constructing dataloaders.
"""

from .engine import Engine
from .model import Model
from .data import Data

__all__ = ["Engine", "Model", "Data"]
