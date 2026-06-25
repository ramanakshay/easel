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

                 # ── Mode flags ──
                 do_train: bool = True,
                 do_val: bool = True,
                 do_test: bool = True,
                 do_predict: bool = True,

                 # ── Loop limits ──
                 max_epochs: Optional[int] = None,
                 max_steps: Optional[int] = None,
                 train_steps_per_epoch: Optional[int] = None,
                 val_steps_per_epoch: Optional[int] = None,
                 test_steps_per_epoch: Optional[int] = None,
                 predict_steps_per_epoch: Optional[int] = None,

                 # ── Validation ──
                 val_strategy: str = "epoch",
                 val_start: int = 0,
                 val_interval: int = 1,

                 # ── Tracker ──
                 project_dir: str = "outputs",
                 project_name: str = "outputs",
                 log_with: Union[str, List[str], None] = None,
                 init_trackers_config: Optional[Dict[str, Any]] = None,

                 # ── Data ──
                 stage: Optional[str] = None,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 dataloader_config: Optional[Dict[str, Any]] = None,

                 # ── Model ──
                 optimizers_config: Optional[Dict[str, Any]] = None,

                 # ── Accelerator ──
                 gradient_accumulation_steps: int = 1,
                 gradient_clip_value: Optional[float] = None,
                 gradient_clip_algorithm: str = "norm",
                 mixed_precision: str = "no",
                 compile: bool = False,
                 sync_batch_norm: bool = False,
                 accelerator_config: Optional[Dict[str, Any]] = None,

                 # ── Reproducibility ──
                 seed: Optional[int] = None,
                 deterministic: bool = False,
                 tf32: Union[bool, str] = False,
                 cudnn_benchmark: bool = False,
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

        self.stage = stage

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_value = gradient_clip_value
        self.gradient_clip_algorithm = gradient_clip_algorithm.lower()

        self.mixed_precision = mixed_precision
        self.compile = compile

        self.sync_batch_norm = sync_batch_norm

        self.seed = seed
        self.deterministic = deterministic
        self.tf32 = tf32
        self.cudnn_benchmark = cudnn_benchmark

        self.project_dir = project_dir
        self.log_with = log_with
        self.project_name = project_name

        self.dataloader_config = dataloader_config or {}
        self.optimizers_config = optimizers_config or {}
        self.accelerator_config = accelerator_config or {}
        self.init_trackers_config = init_trackers_config or {}

        # TODO: think of the best way to manage config key + named arguments
        named_tracker_args = {"project_name": self.project_name}
        overlap = set(self.init_trackers_config.keys()) & set(named_tracker_args.keys())
        if overlap:
            logger.warning(
                f"init_trackers_config keys {overlap} overlap with named arguments. "
                f"init_trackers_config values take precedence."
            )
        self.init_trackers_config = {**named_tracker_args, **self.init_trackers_config}
        self.optimizers = []
        self.schedulers = []
        self.monitor = {}
        self.should_stop = False

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.predict_dataloader = None

        self.setup_globals()
        self.setup_accelerator()
        self.setup_data()
        self.setup_model()


    # ------------------------------------------------------------------
    # Setup: globals
    # ------------------------------------------------------------------

    def setup_globals(self):
        if self.deterministic:
            if self.cudnn_benchmark:
                logger.warning("cudnn_benchmark cannot be True if deterministic is True. Disabling benchmark.")
                self.cudnn_benchmark = False

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True

        if self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        if self.tf32:
            precision = self.tf32 if isinstance(self.tf32, str) else "high"
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    # Setup: accelerator
    # ------------------------------------------------------------------

    def setup_accelerator(self):
        named_accelerator_args = {
            "project_dir": self.project_dir,
            "log_with": self.log_with,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
        }

        overlap = set(self.accelerator_config.keys()) & set(named_accelerator_args.keys())
        if overlap:
            logger.warning(
                f"accelerator_config keys {overlap} overlap with named arguments. "
                f"accelerator_config values take precedence."
            )

        accelerator_kwargs = named_accelerator_args.copy()
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
        if self.seed is not None:
            set_seed(self.seed, device_specific=True)

        if self.log_with and self.accelerator.is_main_process:
            init_trackers_kwargs = self.init_trackers_config.copy()
            project_name = init_trackers_kwargs.pop("project_name")
            config = init_trackers_kwargs.pop("config", None)
            init_kwargs = init_trackers_kwargs.pop("init_kwargs", None)
            self.accelerator.init_trackers(project_name, config=config, init_kwargs=init_kwargs)

    # ------------------------------------------------------------------
    # Setup: data
    # ------------------------------------------------------------------

    def setup_data(self):
        if self.data is None:
            return

        if self.accelerator.is_main_process:
            self.data.prepare()
        self.accelerator.wait_for_everyone()

        self.data.setup(stage=self.stage)

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

                if loader is None:
                    setattr(self, f"do_{mode}", False)

        if loaders_to_prepare:
            prepared_loaders = self.accelerator.prepare(*loaders_to_prepare)

            if not isinstance(prepared_loaders, tuple):
                prepared_loaders = (prepared_loaders,)

            for i, mode in enumerate(modes):
                setattr(self, f"{mode}_dataloader", prepared_loaders[i])

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
                num = len(loader)
                if mode == "train":
                    num = math.ceil(num / self.gradient_accumulation_steps)
                setattr(self, attr_name, num)
            except TypeError:
                pass

        if self.do_train:
            if self.max_epochs is None and self.max_steps is None:
                raise ValueError("At least one of max_epochs or max_steps must be specified.")
            if self.max_steps is None and self.max_epochs is not None:
                if self.train_steps_per_epoch is not None:
                    self.max_steps = self.max_epochs * self.train_steps_per_epoch
            if self.max_epochs is None and self.max_steps is not None:
                if self.train_steps_per_epoch is not None:
                    self.max_epochs = math.ceil(self.max_steps / self.train_steps_per_epoch)

    def _get_dataloader_kwargs(self, mode: str) -> Dict[str, Any]:
        kwargs = {}

        if not self.dataloader_config:
            return {
                "batch_size": self.train_batch_size if mode == "train" else self.eval_batch_size
            }

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

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self.train_batch_size if mode == "train" else self.eval_batch_size

        return kwargs

    def _fetch_loader(self, mode: str, kwargs: Dict[str, Any]) -> Any:
        method_name = f"{mode}_dataloader"
        method = getattr(self.data, method_name)

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

        configure_optim = self.model.configure_optimizers

        sig = inspect.signature(configure_optim)
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

        if accepts_kwargs:
            opt_conf = configure_optim(**self.optimizers_config)
        else:
            valid_kwargs = {k: v for k, v in self.optimizers_config.items() if k in sig.parameters}
            dropped_keys = set(self.optimizers_config.keys()) - set(valid_kwargs.keys())
            if dropped_keys:
                logger.debug(
                    f"Ignored kwargs for `configure_optimizers` because they are not in the signature: {dropped_keys}. "
                    f"To use them, add `**kwargs` to your method definition."
                )
            opt_conf = configure_optim(**valid_kwargs)

        self._standardize_optimizers(opt_conf)

        if not self.optimizers:
            raise ValueError(
                "do_train=True requires at least one optimizer from configure_optimizers."
            )

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

        if opt_conf is None:
            pass

        elif isinstance(opt_conf, torch.optim.Optimizer):
            raw_optimizers = [opt_conf]

        elif isinstance(opt_conf, dict):
            if "optimizer" not in opt_conf:
                raise ValueError("Dict must contain an 'optimizer' key.")
            raw_optimizers = [opt_conf["optimizer"]]
            scheduler = opt_conf.get("lr_scheduler") or opt_conf.get("scheduler")
            if scheduler is not None:
                raw_schedulers = [scheduler]

        elif isinstance(opt_conf, (list, tuple)):
            if len(opt_conf) == 2 and isinstance(opt_conf[0], (list, tuple)):
                raw_optimizers = list(opt_conf[0])
                scheds = opt_conf[1]
                raw_schedulers = list(scheds) if isinstance(scheds, (list, tuple)) else [scheds]

            elif len(opt_conf) > 0 and isinstance(opt_conf[0], dict):
                for d in opt_conf:
                    if "optimizer" not in d:
                        raise ValueError("Each dict must contain an 'optimizer' key.")
                    raw_optimizers.append(d["optimizer"])
                    scheduler = d.get("lr_scheduler") or d.get("scheduler")
                    if scheduler is not None:
                        raw_schedulers.append(scheduler)

            elif len(opt_conf) == 2 and isinstance(opt_conf[0], torch.optim.Optimizer):
                raw_optimizers = [opt_conf[0]]
                sched = opt_conf[1]
                raw_schedulers = [sched] if not isinstance(sched, list) else sched

            else:
                raw_optimizers = list(opt_conf)

        else:
            raise TypeError(
                f"Unsupported return type from configure_optimizers: {type(opt_conf).__name__}"
            )

        for i, opt in enumerate(raw_optimizers):
            if not isinstance(opt, torch.optim.Optimizer):
                raise TypeError(f"Expected Optimizer at index {i}, got {type(opt).__name__}")
            self.optimizers.append(opt)

        for sched_item in raw_schedulers:
            if sched_item is None:
                continue

            std_sched = {
                'scheduler': None,
                'strategy': 'epoch',
                'interval': 1,
                'monitor': None,
            }

            if isinstance(sched_item, dict):
                if 'scheduler' not in sched_item:
                    raise ValueError("Scheduler config dict must contain a 'scheduler' key.")
                std_sched.update(sched_item)
            else:
                std_sched['scheduler'] = sched_item

            if isinstance(std_sched['scheduler'], torch.optim.lr_scheduler.ReduceLROnPlateau):
                if std_sched['monitor'] is None:
                    raise ValueError(
                        "ReduceLROnPlateau requires a 'monitor' key in the scheduler config."
                    )

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

    @property
    def use_distributed(self) -> bool:
        return self.accelerator.use_distributed

    @property
    def local_process_index(self) -> int:
        return self.accelerator.local_process_index

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

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

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
        if self.gradient_clip_value is not None:
            if self.gradient_clip_algorithm == "value":
                self.accelerator.clip_grad_value_(self.model.parameters(), self.gradient_clip_value)
            else:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)

    def optimizer_zero_grad(self, idx: int, set_to_none: bool = True):
        self.optimizers[idx].zero_grad(set_to_none=set_to_none)

    def optimizers_zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def optimizer_step(self, idx: int):
        self.optimizers[idx].step()

    def optimizers_step(self):
        for opt in self.optimizers:
            opt.step()

    def scheduler_step(self, idx: int):
        sched_dict = self.schedulers[idx]
        monitor_key = sched_dict['monitor']

        if monitor_key is not None:
            if monitor_key not in self.monitor:
                raise KeyError(
                    f"Scheduler expected '{monitor_key}' in `engine.monitor`, but it was not found. "
                    f"Make sure to populate `engine.monitor['{monitor_key}']` before stepping."
                )
            sched_dict['scheduler'].step(self.monitor[monitor_key])
        else:
            sched_dict['scheduler'].step()

    def schedulers_step(self, strategy: str):
        counter = self.step if strategy == "step" else self.epoch
        for i, sched_dict in enumerate(self.schedulers):
            if sched_dict['strategy'] == strategy:
                if counter % sched_dict['interval'] == 0:
                    self.scheduler_step(i)

    # ------------------------------------------------------------------
    # Loop: run (orchestrator)
    # ------------------------------------------------------------------

    def run(self):
        if self.do_train:
            self.run_train()
        elif self.do_val:
            self.run_val()
        if self.do_test:
            self.run_test()
        if self.do_predict:
            self.run_predict()

    def run_train(self):
        self.on_train_start()

        epoch_idx = 0
        while self.max_epochs is None or epoch_idx < self.max_epochs:
            self.model.train()
            self.on_train_epoch_start()

            epoch_completed = True
            has_batch = False
            for batch_idx, batch in enumerate(self.train_dataloader):
                has_batch = True
                with self.accumulate():
                    self.on_train_substep_start(batch, batch_idx)
                    loss = self.train_step(batch)
                    self.backward(loss) # TODO: Also support dictionaries 'loss' key
                    self.on_train_substep_end(loss, batch, batch_idx)

                    if self.sync_gradients:
                        self.clip_gradients()

                if self.sync_gradients:
                    self.on_train_step_start()
                    self.optimizers_step()
                    self.optimizers_zero_grad()
                    self.step += 1
                    self.schedulers_step(strategy="step")
                    self.on_train_step_end()

                    if self.val_strategy == "step" and self.should_validate():
                        self.run_val()
                        self.model.train()

                    if self.max_steps is not None and self.step >= self.max_steps:
                        epoch_completed = False
                        break

            if not has_batch and self.max_epochs is None:
                logger.warning("Dataloader produced no batches. Stopping training.")
                break

            if epoch_completed:
                self.on_train_epoch_end()
                self.epoch += 1
                self.schedulers_step(strategy="epoch")

                if self.val_strategy == "epoch" and self.should_validate():
                    self.run_val()
                    self.model.train()

            if self.max_steps is not None and self.step >= self.max_steps:
                break

            epoch_idx += 1

        self.on_train_end()

    def run_val(self):
        if self.val_dataloader is None:
            return
        self.model.eval()
        self.on_val_start()
        self.on_val_epoch_start()
        for batch_idx, batch in enumerate(self.val_dataloader):
            if self.val_steps_per_epoch is not None and batch_idx >= self.val_steps_per_epoch:
                break
            self.on_val_step_start(batch, batch_idx)
            with torch.no_grad():
                outputs = self.val_step(batch)
            self.on_val_step_end(outputs, batch, batch_idx)
        self.on_val_epoch_end()
        self.on_val_end()

    def run_test(self):
        if self.test_dataloader is None:
            return
        self.model.eval()
        self.on_test_start()
        self.on_test_epoch_start()
        for batch_idx, batch in enumerate(self.test_dataloader):
            if self.test_steps_per_epoch is not None and batch_idx >= self.test_steps_per_epoch:
                break
            self.on_test_step_start(batch, batch_idx)
            with torch.no_grad():
                outputs = self.test_step(batch)
            self.on_test_step_end(outputs, batch, batch_idx)
        self.on_test_epoch_end()
        self.on_test_end()

    def run_predict(self):
        if self.predict_dataloader is None:
            return
        self.model.eval()
        self.on_predict_start()
        self.on_predict_epoch_start()
        for batch_idx, batch in enumerate(self.predict_dataloader):
            if self.predict_steps_per_epoch is not None and batch_idx >= self.predict_steps_per_epoch:
                break
            self.on_predict_step_start(batch, batch_idx)
            with torch.no_grad():
                outputs = self.predict_step(batch)
            self.on_predict_step_end(outputs, batch, batch_idx)
        self.on_predict_epoch_end()
        self.on_predict_end()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def should_validate(self) -> bool:
        if not self.do_val or self.val_strategy == "no":
            return False
        counter = self.step if self.val_strategy == "step" else self.epoch
        return counter >= self.val_start and counter % self.val_interval == 0

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
    def on_train_substep_start(self, batch, batch_idx): pass
    def on_train_substep_end(self, outputs, batch, batch_idx): pass
    def on_train_step_start(self): pass
    def on_train_step_end(self): pass
    def on_train_epoch_end(self): pass
    def on_train_end(self): pass

    def on_val_start(self): pass
    def on_val_epoch_start(self): pass
    def on_val_step_start(self, batch, batch_idx): pass
    def on_val_step_end(self, outputs, batch, batch_idx): pass
    def on_val_epoch_end(self): pass
    def on_val_end(self): pass

    def on_test_start(self): pass
    def on_test_epoch_start(self): pass
    def on_test_step_start(self, batch, batch_idx): pass
    def on_test_step_end(self, outputs, batch, batch_idx): pass
    def on_test_epoch_end(self): pass
    def on_test_end(self): pass

    def on_predict_start(self): pass
    def on_predict_epoch_start(self): pass
    def on_predict_step_start(self, batch, batch_idx): pass
    def on_predict_step_end(self, outputs, batch, batch_idx): pass
    def on_predict_epoch_end(self): pass
    def on_predict_end(self): pass
