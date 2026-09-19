# Easel

[![PyPI version](https://badge.fury.io/py/easel.svg)](https://pypi.org/project/easel/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Easel is a modular, flexible, and scalable deep learning framework built on top of PyTorch and HuggingFace Accelerate. It organizes deep learning projects into three logically separate modules — Data, Model, and Engine — providing a clean structure without heavy boilerplate. Distributed training, mixed precision, gradient accumulation, and experiment tracking are all natively supported through Accelerate.

---

## Installation

Pip users:

```bash
pip install easel
```

Uv users:

```bash
uv add easel
```

**Requirements:** Python 3.8+, PyTorch 2.0+, [accelerate](https://github.com/huggingface/accelerate).

---

## Tutorial

A supervised learning pipeline in Easel consists of three modules:

- **Data** — Handles dataset construction and dataloader creation.
- **Model** — Defines the network architecture and optimizer configuration.
- **Engine** — Orchestrates the training, validation, testing, and prediction loops.

Each module is independent: the Model knows nothing about training or validation logic, the Data knows nothing about the model, and the Engine ties them together. Let's build each one.

### Data

Subclass `Data` and override `setup` to assign your datasets. Easel calls `setup` automatically and builds dataloaders from whatever datasets you assign.

```python
import torch
from torch.utils.data import TensorDataset
from easel import Data


class MyData(Data):
    def setup(self, stage=None):
        x = torch.randn(1000, 10)
        y = torch.randn(1000, 1)
        ds = TensorDataset(x, y)
        self.train_dataset = ds
        self.val_dataset = ds
```

### Model

Subclass `Model` and define your architecture in `forward`, plus your optimizer and optional LR scheduler in `configure_optimizers`. The Model contains no training or validation logic — it only defines the network and how to optimize it.

```python
import torch.nn as nn
from easel import Model


class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        return {"optimizer": opt, "lr_scheduler": sched}
```

### Engine

Subclass `Engine` and implement `train_step` and `val_step`. These methods receive a batch and return a loss tensor (or a dict with a `"loss"` key). The Engine handles the rest — the training loop, gradient accumulation, optimizer steps, scheduler steps, and validation scheduling.

```python
import torch.nn as nn
from easel import Engine


class MyEngine(Engine):
    def train_step(self, batch):
        x, y = batch
        return nn.functional.mse_loss(self.model(x), y)

    def val_step(self, batch):
        x, y = batch
        return {"val_loss": nn.functional.mse_loss(self.model(x), y)}
```

### Putting it together

Pass your Data and Model to the Engine, configure the training parameters, and call `run`.

```python
engine = MyEngine(
    model=MyModel(),
    data=MyData(),
    train_batch_size=32,
    max_epochs=10,
    seed=42,
)
engine.run()
```

This trains the model with MSE loss, runs validation every epoch, and steps the LR scheduler automatically. To use your own data, override `setup`; to use your own architecture, override `forward` and `configure_optimizers`.

---

## Features

- **Modular design** — Clean separation of logic across `Data`, `Model`, and `Engine`. The Model defines only the network and optimizer, with no details about training or validation. The Data handles only dataset construction. The Engine orchestrates everything.
- **Accelerate integration** — Distributed training (multi-GPU, multi-node), mixed precision (FP16, BF16, FP8), gradient accumulation, and experiment tracking (WandB, TensorBoard) all work out of the box with no extra code.
- **Lifecycle hooks** — Override `on_train_epoch_start`, `on_val_step_end`, and 20+ other hooks to inject custom logic at any point in the training loop without modifying the loop itself.

---

## License

Easel is released under the [MIT License](LICENSE).
