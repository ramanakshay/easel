import torch
import torch.nn as nn
from typing import Any, Union, Dict, Tuple, List

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        return None
