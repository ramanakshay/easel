import os
import inspect
import logging
import gc
import math
from typing import Any, Dict, List, Optional, Union

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, TorchDynamoPlugin

from .data import Data
from .model import Model

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self,
                 data: Data,
                 model: Model,

                 do_train: bool = False,
                 do_val: bool = False,
                 do_test: bool = False,
                 do_predict: bool = False,

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
                 log_config: Optional[Dict[str, Any]] = None,
                 ):

        self.model = model
        self.data = data

        self.do_train = do_train
        self.do_val = do_val
        self.do_test = do_test
        self.do_predict = do_predict

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.train_steps_per_epoch = train_steps_per_epoch
        self.val_steps_per_epoch = val_steps_per_epoch
        self.test_steps_per_epoch = test_steps_per_epoch
        self.predict_steps_per_epoch = predict_steps_per_epoch

        self.step = 0
        self.epoch = 0

        self.val_strategy = val_strategy
        self.val_start = val_start
        self.val_interval = val_interval

        self.optimizer_config = optimizer_config or {}
        self.optimizers = []
        self.schedulers = []
        self.monitor = {}

        self.stage = stage
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataloader_config = dataloader_config or {}
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.predict_dataloader = None

        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.compile = compile
        self.gradient_clip_value = gradient_clip_value
        self.gradient_clip_algorithm = gradient_clip_algorithm.lower()
        self.sync_batch_norm = sync_batch_norm
        self.accelerator_config = accelerator_config or {}

        self.seed = seed
        self.deterministic = deterministic
        self.tf32 = tf32
        self.cudnn_benchmark = cudnn_benchmark

        self.project_dir = project_dir
        self.project_name = project_name
        self.run_config = run_config or {}
        self.log_with = log_with
        self.log_config = log_config or {}

        self.setup_accelerator()
        self.setup_globals()
        self.setup_data()
        self._resolve_steps_per_epoch()
        self._resolve_loop_limits()
        self.setup_model()

    # ------------------------------------------------------------------
    # Setup: accelerator
    # ------------------------------------------------------------------

    def setup_accelerator(self):
        accelerator_kwargs = {
            "project_dir": self.project_dir,
            "log_with": self.log_with,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
        }

        accelerator_kwargs.update(self.accelerator_config)

        if self.compile and "dynamo_plugin" not in accelerator_kwargs:
            dynamo_plugin = TorchDynamoPlugin(
                backend="inductor",
                mode="default",
                fullgraph=False,
                dynamic=False
            )
            accelerator_kwargs["dynamo_plugin"] = dynamo_plugin

        self.accelerator = Accelerator(**accelerator_kwargs)

        if self.log_with and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.project_name,
                config=self.run_config,
                init_kwargs=self.log_config
            )

    # ------------------------------------------------------------------
    # Setup: globals
    # ------------------------------------------------------------------

    def setup_globals(self):
        set_seed(self.seed, device_specific=True)

        if self.deterministic:
            if self.cudnn_benchmark:
                logger.warning("cudnn_benchmark cannot be True if deterministic is True. Disabling benchmark.")
                self.cudnn_benchmark = False

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)

        if self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        if self.tf32:
            precision = self.tf32 if isinstance(self.tf32, str) else "high"
            torch.set_float32_matmul_precision(precision)

    # ------------------------------------------------------------------
    # Setup: data
    # ------------------------------------------------------------------

    def setup_data(self):
        if self.data is None:
            return

        if hasattr(self.data, "prepare"):
            if self.accelerator.is_main_process:
                self.data.prepare()
            self.accelerator.wait_for_everyone()

        if hasattr(self.data, "setup"):
            sig = inspect.signature(self.data.setup)
            if 'stage' in sig.parameters:
                self.data.setup(stage=self.stage)
            else:
                self.data.setup()

        loaders_to_prepare = []
        modes = []

        for mode in ["train", "val", "test", "predict"]:
            if getattr(self, f"do_{mode}"):
                kwargs = self._get_dataloader_kwargs(mode)
                loader = self._fetch_loader(mode, kwargs)

                setattr(self, f"{mode}_dataloader", loader)

                if loader is not None:
                    loaders_to_prepare.append(loader)
                    modes.append(mode)

        if loaders_to_prepare:
            prepared_loaders = self.accelerator.prepare(*loaders_to_prepare)

            if not isinstance(prepared_loaders, tuple):
                prepared_loaders = (prepared_loaders,)

            for i, mode in enumerate(modes):
                setattr(self, f"{mode}_dataloader", prepared_loaders[i])

    def _get_dataloader_kwargs(self, mode: str) -> Dict[str, Any]:
        kwargs = {
            "batch_size": self.train_batch_size if mode == "train" else self.eval_batch_size
        }

        if not self.dataloader_config:
            return kwargs

        mode_prefixes = ("train_", "val_", "test_", "predict_")
        for k, v in self.dataloader_config.items():
            if not isinstance(v, dict) and not k.startswith(mode_prefixes):
                kwargs[k] = v

        prefix = f"{mode}_"
        for k, v in self.dataloader_config.items():
            if k.startswith(prefix):
                kwargs[k[len(prefix):]] = v

        section = self.dataloader_config.get(mode)
        if isinstance(section, dict):
            kwargs.update(section)

        return kwargs

    def _fetch_loader(self, mode: str, kwargs: Dict[str, Any]) -> Any:
        method_name = f"{mode}_dataloader"
        method = getattr(self.data, method_name, None)

        if not method:
            return None

        sig = inspect.signature(method)

        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

        if accepts_kwargs:
            return method(**kwargs)
        else:
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

            dropped_keys = set(kwargs.keys()) - set(valid_kwargs.keys())
            if dropped_keys:
                logger.debug(
                    f"Ignored kwargs for `{method_name}` because they are not in the signature: {dropped_keys}. "
                    f"To use them, add `**kwargs` to your method definition."
                )

            return method(**valid_kwargs)

    # ------------------------------------------------------------------
    # Setup: resolve steps per epoch + loop limits
    # ------------------------------------------------------------------

    def _resolve_steps_per_epoch(self):
        for mode in ["train", "val", "test", "predict"]:
            if not getattr(self, f"do_{mode}"):
                continue
            attr_name = f"{mode}_steps_per_epoch"
            if getattr(self, attr_name) is not None:
                continue
            loader = getattr(self, f"{mode}_dataloader")
            if loader is None:
                continue
            try:
                setattr(self, attr_name, len(loader))
            except TypeError:
                pass

    def _resolve_loop_limits(self):
        if self.max_epochs is None and self.max_steps is None:
            raise ValueError("At least one of max_epochs or max_steps must be specified.")

        if self.max_steps is None and self.max_epochs is not None:
            if self.train_steps_per_epoch is not None:
                self.max_steps = self.max_epochs * self.train_steps_per_epoch

    # ------------------------------------------------------------------
    # Setup: model + optimizers
    # ------------------------------------------------------------------

    def setup_model(self):
        if not self.do_train:
            logger.info("do_train=False. Skipping optimizers and preparing model for inference.")
            prepared_objs = self.accelerator.prepare(self.model)
            if not isinstance(prepared_objs, tuple):
                prepared_objs = (prepared_objs,)
            self.model = prepared_objs[0]
            return

        if self.sync_batch_norm and self.accelerator.num_processes > 1:
            logger.info("Converting model to SyncBatchNorm...")
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        configure_optim = getattr(self.model, "configure_optimizers", None)
        if not configure_optim:
            raise NotImplementedError("To train, you must implement `configure_optimizers()` in your Module.")

        sig = inspect.signature(configure_optim)
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

        if accepts_kwargs:
            opt_conf = configure_optim(**self.optimizer_config)
        else:
            valid_kwargs = {k: v for k, v in self.optimizer_config.items() if k in sig.parameters}

            dropped_keys = set(self.optimizer_config.keys()) - set(valid_kwargs.keys())
            if dropped_keys:
                logger.debug(
                    f"Ignored kwargs for `configure_optimizers` because they are not in the signature: {dropped_keys}. "
                    f"To use them, add `**kwargs` to your method definition."
                )
            opt_conf = configure_optim(**valid_kwargs)

        if opt_conf is None:
            raise ValueError("`configure_optimizers` returned None, but do_train=True.")

        self._standardize_optimizers(opt_conf)

        to_prepare = [self.model] + self.optimizers + [s['scheduler'] for s in self.schedulers]

        prepared_objs = self.accelerator.prepare(*to_prepare)

        if not isinstance(prepared_objs, tuple):
            prepared_objs = (prepared_objs,)

        self.model = prepared_objs[0]

        curr_idx = 1
        if self.optimizers:
            self.optimizers = list(prepared_objs[curr_idx: curr_idx + len(self.optimizers)])
            curr_idx += len(self.optimizers)

        if self.schedulers:
            prepared_schedulers = prepared_objs[curr_idx:]
            for i, prep_sched in enumerate(prepared_schedulers):
                self.schedulers[i]['scheduler'] = prep_sched

    def _standardize_optimizers(self, opt_conf: Any):
        raw_optimizers = []
        raw_schedulers = []
        common_config = {}

        if isinstance(opt_conf, dict):
            common_config = opt_conf.copy()
            opt_input = common_config.pop('optimizer', None) or common_config.pop('optimizers', None)
            if opt_input is None:
                raise ValueError("Optimizer config dict must contain an 'optimizer' or 'optimizers' key.")

            raw_optimizers = opt_input if isinstance(opt_input, list) else [opt_input]

            sched_input = common_config.pop('scheduler', None) or common_config.pop('schedulers', None)
            if sched_input is not None:
                raw_schedulers = sched_input if isinstance(sched_input, list) else [sched_input]

        elif isinstance(opt_conf, tuple):
            if len(opt_conf) != 2:
                raise ValueError(f"Tuple config must be (Optimizers, Schedulers). Got length {len(opt_conf)}")
            raw_optimizers = opt_conf[0] if isinstance(opt_conf[0], list) else [opt_conf[0]]
            raw_schedulers = opt_conf[1] if isinstance(opt_conf[1], list) else [opt_conf[1]]

        else:
            raw_optimizers = opt_conf if isinstance(opt_conf, list) else [opt_conf]

        for i, opt in enumerate(raw_optimizers):
            if not isinstance(opt, torch.optim.Optimizer):
                raise TypeError(f"Expected torch.optim.Optimizer at index {i}, got {type(opt).__name__}")
            self.optimizers.append(opt)

        for sched_item in raw_schedulers:
            if sched_item is None:
                continue

            std_sched = {
                'scheduler': None,
                'strategy': 'epoch',
                'interval': 1,
                'monitor': None,
                'strict': True,
            }

            if isinstance(sched_item, dict):
                if 'scheduler' not in sched_item:
                    raise ValueError("Scheduler dictionary must contain a 'scheduler' key")
                std_sched.update(common_config)
                std_sched.update(sched_item)
            else:
                std_sched.update(common_config)
                std_sched['scheduler'] = sched_item

            self.schedulers.append(std_sched)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def raw_model(self) -> torch.nn.Module:
        return self.accelerator.unwrap_model(self.model)

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def is_main_process(self) -> bool:
        return self.accelerator.is_main_process

    @property
    def num_processes(self) -> int:
        return self.accelerator.num_processes

    @property
    def sync_gradients(self) -> bool:
        return self.accelerator.sync_gradients

    # ------------------------------------------------------------------
    # Context managers + distributed utils
    # ------------------------------------------------------------------

    def autocast(self):
        return self.accelerator.autocast()

    def accumulate(self):
        return self.accelerator.accumulate(self.model)

    def wait(self):
        self.accelerator.wait_for_everyone()

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def log(self, values: Dict[str, Any], step: Optional[int] = None):
        self.accelerator.log(values, step=step)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.accelerator.gather(tensor)

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.accelerator.gather_for_metrics(tensor)

    def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        return self.accelerator.reduce(tensor, reduction=reduction)

    def free_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Training primitives
    # ------------------------------------------------------------------

    def backward(self, loss: torch.Tensor, **kwargs):
        self.accelerator.backward(loss, **kwargs)

    def clip_gradients(self):
        if self.sync_gradients and self.gradient_clip_value > 0.0:
            if self.gradient_clip_algorithm == "value":
                self.accelerator.clip_grad_value_(self.model.parameters(), self.gradient_clip_value)
            else:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)

    def optimizers_zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def optimizers_step(self):
        for opt in self.optimizers:
            opt.step()

    def schedulers_step(self, strategy: str, counter: int):
        for sched_dict in self.schedulers:
            if sched_dict['strategy'] == strategy:
                if counter % sched_dict['interval'] == 0:
                    self._execute_scheduler_step(sched_dict)

    def _execute_scheduler_step(self, sched_dict: Dict[str, Any]):
        scheduler = sched_dict['scheduler']
        monitor_key = sched_dict['monitor']

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if monitor_key not in self.monitor:
                if sched_dict['strict']:
                    raise KeyError(
                        f"Scheduler expected '{monitor_key}' in `engine.monitor`, but it was not found. "
                        f"Set strict=False to ignore, or populate `engine.monitor['{monitor_key}']` before stepping."
                    )
                else:
                    logger.debug(f"Skipping scheduler step: '{monitor_key}' missing from monitor.")
                    return

            scheduler.step(self.monitor[monitor_key])
        else:
            scheduler.step()

    # ------------------------------------------------------------------
    # Loop: run (orchestrator)
    # ------------------------------------------------------------------

    def run(self):
        if self.do_train:
            self.run_train()
        if self.do_val:
            self.run_val()
        if self.do_test:
            self.run_test()
        if self.do_predict:
            self.run_predict()

    def run_train(self):
        self.on_train_start()

        epoch_idx = 0
        while self.max_epochs is None or epoch_idx < self.max_epochs:
            self.on_train_epoch_start()

            for batch in self.train_dataloader:
                with self.accumulate():
                    self.optimizers_zero_grad()
                    self.on_train_substep_start()
                    loss = self.train_step(batch)
                    self.backward(loss)
                    self.on_train_substep_end()

                    if self.sync_gradients:
                        self.clip_gradients()

                if self.sync_gradients:
                    self.on_train_step_start()
                    self.optimizers_step()
                    self.optimizers_zero_grad()
                    self.step += 1
                    self.schedulers_step(strategy="step", counter=self.step)
                    self.on_train_step_end()

                    if self.val_strategy == "step" and self.should_validate():
                        self.run_val()

                    if self.max_steps is not None and self.step >= self.max_steps:
                        break

            self.on_train_epoch_end()
            self.epoch += 1
            self.schedulers_step(strategy="epoch", counter=self.epoch)

            if self.val_strategy == "epoch" and self.should_validate():
                self.run_val()

            if self.max_steps is not None and self.step >= self.max_steps:
                break

            epoch_idx += 1

        self.on_train_end()

    def run_val(self):
        self.on_val_start()
        self.on_val_epoch_start()
        for batch_idx, batch in enumerate(self.val_dataloader):
            if self.val_steps_per_epoch is not None and batch_idx >= self.val_steps_per_epoch:
                break
            self.on_val_step_start()
            metrics = self.val_step(batch)
            if metrics:
                self.monitor.update(metrics)
            self.on_val_step_end()
        self.on_val_epoch_end()
        self.on_val_end()

    def run_test(self):
        self.on_test_start()
        self.on_test_epoch_start()
        for batch_idx, batch in enumerate(self.test_dataloader):
            if self.test_steps_per_epoch is not None and batch_idx >= self.test_steps_per_epoch:
                break
            self.on_test_step_start()
            metrics = self.test_step(batch)
            if metrics:
                self.monitor.update(metrics)
            self.on_test_step_end()
        self.on_test_epoch_end()
        self.on_test_end()

    def run_predict(self):
        self.on_predict_start()
        self.on_predict_epoch_start()
        for batch_idx, batch in enumerate(self.predict_dataloader):
            if self.predict_steps_per_epoch is not None and batch_idx >= self.predict_steps_per_epoch:
                break
            self.on_predict_step_start()
            self.predict_step(batch)
            self.on_predict_step_end()
        self.on_predict_epoch_end()
        self.on_predict_end()

    # ------------------------------------------------------------------
    # Step methods (user implements)
    # ------------------------------------------------------------------

    def train_step(self, batch) -> torch.Tensor:
        raise NotImplementedError("train_step must be implemented to train.")

    def val_step(self, batch) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("val_step must be implemented to validate.")

    def test_step(self, batch) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("test_step must be implemented to test.")

    def predict_step(self, batch) -> Any:
        raise NotImplementedError("predict_step must be implemented to predict.")

    # ------------------------------------------------------------------
    # Lifecycle hooks (all no-ops by default)
    # ------------------------------------------------------------------

    def on_train_start(self): pass
    def on_train_epoch_start(self): pass
    def on_train_substep_start(self): pass
    def on_train_substep_end(self): pass
    def on_train_step_start(self): pass
    def on_train_step_end(self): pass
    def on_train_epoch_end(self): pass
    def on_train_end(self): pass

    def on_val_start(self): pass
    def on_val_epoch_start(self): pass
    def on_val_step_start(self): pass
    def on_val_step_end(self): pass
    def on_val_epoch_end(self): pass
    def on_val_end(self): pass

    def on_test_start(self): pass
    def on_test_epoch_start(self): pass
    def on_test_step_start(self): pass
    def on_test_step_end(self): pass
    def on_test_epoch_end(self): pass
    def on_test_end(self): pass

    def on_predict_start(self): pass
    def on_predict_epoch_start(self): pass
    def on_predict_step_start(self): pass
    def on_predict_step_end(self): pass
    def on_predict_epoch_end(self): pass
    def on_predict_end(self): pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def should_validate(self) -> bool:
        if not self.do_val:
            return False
        counter = self.step if self.val_strategy == "step" else self.epoch
        return counter >= self.val_start and counter % self.val_interval == 0
