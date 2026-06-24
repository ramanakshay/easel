import torch
import torch.nn as nn
from typing import Any, Union, Dict, Tuple, List

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        raise NotImplementedError(
            "configure_optimizers must be implemented in your Model subclass."
        )
