"""Easel Model Class

This module defines the :class:`Model` base class, which users subclass to
define their network architecture and optimizer configuration.

Public objects:
    Model: Base class for Easel models.
"""

import torch
import torch.nn as nn
from typing import Any, Union, Dict, Tuple, List


class Model(nn.Module):
    """Base class for Easel models.

    Subclass :class:`Model` to define a network architecture in
    :meth:`forward` and optimizers/schedulers in :meth:`configure_optimizers`. The latter supports several
    flexible return formats; see :meth:`configure_optimizers` for details.

    A minimal example::

        class LinearModel(Model):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 1)

            def forward(self, x):
                return self.layer(x)

            def configure_optimizers(self):
                opt = torch.optim.Adam(self.parameters(), lr=1e-3)
                sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
                return {"optimizer": opt, "lr_scheduler": sched}
    """

    def __init__(self) -> None:
        super().__init__()

    def configure_optimizers(self) -> Union[
        torch.optim.Optimizer,
        Dict[str, Any],
        List[Dict[str, Any]],
        Tuple,
        None,
    ]:
        """Configure and return optimizers and optional schedulers.

        It may return any of the following formats:

        1. A single :class:`~torch.optim.Optimizer`.
        2. ``None`` (only valid when ``do_train=False``).
        3. A dict with an ``"optimizer"`` key and an optional
           ``"scheduler"`` (or ``"lr_scheduler"``) key. The scheduler may be
           a bare scheduler or a config dict carrying ``scheduler``,
           ``strategy`` (``"epoch"`` or ``"step"``), ``interval``, and
           ``monitor``.
        4. A list of such dicts (for multiple optimizers).
        5. A tuple ``(optimizers, schedulers)`` where each element is a
           single optimizer/scheduler or a list thereof.
        6. A plain list of optimizers (no schedulers).

        Returns:
            The optimizer configuration in one of the supported formats.

        Raises:
            NotImplementedError: If the subclass does not override this
                method.
        """
        raise NotImplementedError(
            "configure_optimizers must be implemented in your Model subclass."
        )
