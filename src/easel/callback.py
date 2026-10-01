"""Easel Callback Class

This module defines the :class:`Callback` base class, which users subclass
to observe and lightly steer the engine loop without touching its core
logic.

Public objects:
    Callback: Base class for engine lifecycle callbacks.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .engine import Engine


class Callback:
    """Base class for callbacks attached to an :class:`~easel.engine.Engine`.

    Subclass :class:`Callback`, override only the hooks you need (every hook
    is a no-op by default), and pass the instance (or a list of instances) to
    ``Engine(callbacks=...)``. The engine invokes each hook automatically at
    the corresponding lifecycle point: never call hooks manually, and do not
    rely on other callbacks or on the order of callbacks in the list.

    At initialization, ``self.engine`` is ``None``;
    ``Engine.setup_callbacks`` assigns the engine before any hook fires, so
    inside hooks you read engine state (``self.engine.step``,
    ``self.engine.model``, ...) from :attr:`engine`. Callbacks must not mix
    into the engine's core logic; ``self.engine`` should be treated as
    read-only, except for one sanctioned mutation: setting
    ``self.engine.should_stop = True`` stops training after the current
    batch or epoch.

    Attributes:
        engine: The :class:`~easel.engine.Engine` this callback is attached
            to (``None`` until the engine attaches itself).

    A minimal example::

        class StepLoggerCallback(Callback):
            def on_optimizer_step_end(self) -> None:
                self.engine.log({"global_step": self.engine.step})
    """

    def __init__(self) -> None:
        """Initialize the callback (unattached).

        Subclasses that override ``__init__`` should call
        ``super().__init__()`` so :attr:`engine` exists. The engine also
        assigns :attr:`engine` unconditionally, so a missing call is
        tolerated until the engine attaches itself.
        """
        self.engine: Optional["Engine"] = None

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Called once at the beginning of training, before the first epoch."""
        pass

    def on_train_epoch_start(self) -> None:
        """Called at the beginning of each training epoch."""
        pass

    def on_train_step_start(self, batch, batch_idx) -> None:
        """Called at the start of each training batch (a gradient-accumulation micro-batch).

        Args:
            batch: The current micro-batch.
            batch_idx: Index of the current micro-batch within the epoch.
        """
        pass

    def on_train_step_end(self, outputs, batch, batch_idx) -> None:
        """Called at the end of each training batch (a gradient-accumulation micro-batch).

        Args:
            outputs: The full return value of the engine's ``train_step``
                (a loss tensor or a dict containing a ``"loss"`` key).
            batch: The current micro-batch.
            batch_idx: Index of the current micro-batch within the epoch.
        """
        pass

    def on_optimizer_step_start(self) -> None:
        """Called at the start of each optimizer step (after gradient sync)."""
        pass

    def on_optimizer_step_end(self) -> None:
        """Called at the end of each optimizer step (after gradient sync)."""
        pass

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch.

        When the hook fires, ``self.engine.epoch`` has already been
        incremented, so it holds the count of completed epochs (1-based).
        """
        pass

    def on_train_end(self) -> None:
        """Called once at the end of training, after the last epoch."""
        pass

    # ------------------------------------------------------------------
    # Validation hooks
    # ------------------------------------------------------------------

    def on_val_start(self) -> None:
        """Called at the beginning of the validation loop."""
        pass

    def on_val_epoch_start(self) -> None:
        """Called at the beginning of each validation epoch (one dataloader pass)."""
        pass

    def on_val_step_start(self, batch, batch_idx) -> None:
        """Called before each validation batch.

        Args:
            batch: The current validation batch.
            batch_idx: Index of the current batch within the validation loop.
        """
        pass

    def on_val_step_end(self, outputs, batch, batch_idx) -> None:
        """Called after each validation batch.

        Args:
            outputs: The return value of the engine's ``val_step``.
            batch: The current validation batch.
            batch_idx: Index of the current batch within the validation loop.
        """
        pass

    def on_val_epoch_end(self) -> None:
        """Called at the end of each validation epoch (one dataloader pass)."""
        pass

    def on_val_end(self) -> None:
        """Called at the end of the validation loop."""
        pass

    # ------------------------------------------------------------------
    # Testing hooks
    # ------------------------------------------------------------------

    def on_test_start(self) -> None:
        """Called at the beginning of the testing loop."""
        pass

    def on_test_epoch_start(self) -> None:
        """Called at the beginning of each test epoch (one dataloader pass)."""
        pass

    def on_test_step_start(self, batch, batch_idx) -> None:
        """Called before each test batch.

        Args:
            batch: The current test batch.
            batch_idx: Index of the current batch within the testing loop.
        """
        pass

    def on_test_step_end(self, outputs, batch, batch_idx) -> None:
        """Called after each test batch.

        Args:
            outputs: The return value of the engine's ``test_step``.
            batch: The current test batch.
            batch_idx: Index of the current batch within the testing loop.
        """
        pass

    def on_test_epoch_end(self) -> None:
        """Called at the end of each test epoch (one dataloader pass)."""
        pass

    def on_test_end(self) -> None:
        """Called at the end of the testing loop."""
        pass

    # ------------------------------------------------------------------
    # Prediction hooks
    # ------------------------------------------------------------------

    def on_predict_start(self) -> None:
        """Called at the beginning of the prediction loop."""
        pass

    def on_predict_epoch_start(self) -> None:
        """Called at the beginning of each prediction epoch (one dataloader pass)."""
        pass

    def on_predict_step_start(self, batch, batch_idx) -> None:
        """Called before each prediction batch.

        Args:
            batch: The current prediction batch.
            batch_idx: Index of the current batch within the prediction loop.
        """
        pass

    def on_predict_step_end(self, outputs, batch, batch_idx) -> None:
        """Called after each prediction batch.

        Args:
            outputs: The return value of the engine's ``predict_step``.
            batch: The current prediction batch.
            batch_idx: Index of the current batch within the prediction loop.
        """
        pass

    def on_predict_epoch_end(self) -> None:
        """Called at the end of each prediction epoch (one dataloader pass)."""
        pass

    def on_predict_end(self) -> None:
        """Called at the end of the prediction loop."""
        pass
