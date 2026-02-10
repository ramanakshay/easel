import torch
import torch.nn as nn
from typing import Any, Optional, Union, List, Tuple

class Module(nn.Module):
    def __init__(self, model: nn.Module = None, optimizers = None, schedulers = None):
        super().__init__()

        # Public attributes
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers

    def forward(self, *args, **kwargs) -> Any:
        """
        Standard PyTorch forward pass.
        """
        if self.model is not None:
            return self.model(*args, **kwargs)

        raise NotImplementedError(
            "Module initialized without a sub-model (self.model is None). "
            "You must either pass a model to __init__ or override forward()."
        )

    def configure_optimizers(self) -> Any:
        """
        Default implementation checks public attributes.
        """
        if self.optimizers is not None:
            if self.schedulers:
                return (self.optimizers, self.schedulers)
            return self.optimizers
        return None
