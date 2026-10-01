"""Easel: a modular, generalizable deep learning library for SL tasks.

Easel provides a lightweight training and inference framework built on top
of PyTorch and HuggingFace ``accelerate``. Users subclass :class:`Model` and
:class:`Data`, implement a handful of ``*_step`` methods on an
:class:`Engine` subclass, pass :class:`Callback` instances for custom
lifecycle behavior, and call :meth:`Engine.run`.

Public objects:
    Engine: Orchestrates the training, validation, testing, and prediction
        loops.
    Model: Base class for user-defined network architectures and optimizer
        configuration.
    Data: Base class for preparing datasets and constructing dataloaders.
    Callback: Base class for observing and steering the engine loop.
"""

from .engine import Engine
from .model import Model
from .data import Data
from .callback import Callback

__all__ = ["Engine", "Model", "Data", "Callback"]
