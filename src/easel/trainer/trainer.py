from typing import Any, Dict, List, Optional, Union
import math

from ..data import Data
from ..model import Model
from .base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self,
                 data: Data,
                 model: Model,
                 do_train: bool = True,
                 do_val: bool = True,
                 do_test: bool = True,
                 do_predict: bool = True,

                 max_epochs: Optional[int] = None,
                 max_steps: Optional[int] = None,

                 train_steps_per_epoch: Optional[int] = None,
                 val_steps_per_epoch: Optional[int] = None,
                 test_steps_per_epoch: Optional[int] = None,
                 predict_steps_per_epoch: Optional[int] = None,

                 val_strategy: str = "epoch",
                 val_start: int = 0,
                 val_interval: int = 1,

                 stage: Optional[str] = None,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 dataloader_config: Optional[Dict[str, Any]] = None,

                 mixed_precision: str = "no",
                 gradient_accumulation_steps: int = 1,
                 compile: bool = False,
                 gradient_clip_value: float = 0.0,
                 gradient_clip_algorithm: str = "norm",
                 sync_batch_norm: bool = False,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 accelerator_config: Optional[Dict[str, Any]] = None,

                 seed: int = 42,
                 deterministic: bool = False,
                 tf32: Union[bool, str] = False,
                 cudnn_benchmark: bool = False,

                 project_dir: str = "outputs",
                 project_name: str = "",
                 run_config: Optional[Dict[str, Any]] = None,
                 log_with: Union[str, List[str], None] = None,
                 log_config: Optional[Dict[str, Any]] = None):

        super().__init__(data,
                     model,
                     do_train,
                     do_val,
                     do_test,
                     do_predict,

                     stage,
                     train_batch_size,
                     eval_batch_size,
                     dataloader_config,

                     mixed_precision,
                     gradient_accumulation_steps,
                     compile,
                     gradient_clip_value,
                     gradient_clip_algorithm,
                     sync_batch_norm,
                     optimizer_config,
                     accelerator_config,

                     seed,
                     deterministic,
                     tf32,
                     cudnn_benchmark,

                     project_dir,
                     project_name,
                     run_config,
                     log_with,
                     log_config,
                     )

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.epoch = 0
        self.step = 0

        self.train_steps_per_epoch = train_steps_per_epoch

    def run(self):
        self.run_train()
        self.run_val()
        self.run_test()
        self.run_predict()


    def run_train():
        if not do_train:
            return
        for epoch in range(self.max_epochs):
            for step in range(self.train_steps_per_epoch):
                self.run_train_step()
                if self.is_val():
                    self.run_val()
                self.step += 1
            self.epoch += 1

    def run_val(self):
        if not do_val:
            return
        for step in range(self.val_steps_per_epoch):
            self.run_val_step()

    def run_test(self.):
        if not do_test:
            return
        for step in range(self.test_steps_per_epoch):
            self.run_test_step()

    def run_predict(self):
        if not do_predict:
            return
        for step in range(self.predict_steps_per_epoch):
            self.run_predict_step()
