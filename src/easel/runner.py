import inspect
import os
import logging
import random
import gc
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union, Callable
from torch.utils.data import IterableDataset
from accelerate import Accelerator
from accelerate.utils import set_seed

# Internal imports
from .module import Module
from .data_module import DataModule

logger = logging.getLogger(__name__)

class DistWorkerInitFunc:
    """
    Ensures dataloader workers are seeded differently across Ranks and Worker IDs.
    """
    def __init__(self, seed: int, rank: int):
        self.seed = seed
        self.rank = rank

    def __call__(self, worker_id):
        # Unique seed: User_Seed + (Rank * Stride) + Worker_ID
        unique_seed = (self.seed + (self.rank * 10000) + worker_id) % 2**32
        np.random.seed(unique_seed)
        random.seed(unique_seed)
        torch.manual_seed(unique_seed)

class Runner:
    """
    A stateless, boilerplate-free training engine integrating Hugging Face Accelerate.
    Handles device placement, distributed setup, and mixed precision.
    """
    def __init__(self,
                 model: Module,
                 data: DataModule,
                 # --- Flags ---
                 do_train: bool = True,
                 do_val: bool = True,
                 do_test: bool = True,

                 # --- Data Configs ---
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 dataloader_config: Optional[Dict[str, Any]] = None,

                 # --- Model Configs ---
                 optimizer_config: Optional[Dict[str, Any]] = None,

                 # --- Accelerator ---
                 mixed_precision: str = "no",
                 grad_accum_steps: int = 1,
                 accelerator_config: Optional[Dict[str, Any]] = None,

                 # --- Logging ---
                 log_with: Union[str, List[str], None] = None,
                 project_dir: str = "logs",
                 project_name: str = "default_run",
                 run_config: Optional[Dict[str, Any]] = None,
                 log_config: Optional[Dict[str, Any]] = None,

                 # --- Reproducibility ---
                 seed: int = 42,
                 deterministic: bool = False,
                 use_tf32: bool = False,
                 cudnn_benchmark: bool = False,

                 # --- Compilation ---
                 compile: bool = False,
                 compile_config: Optional[Dict[str, Any]] = None,

                 # --- Optimization ---
                 grad_clip_value: float = 0.0,
                 grad_clip_algorithm: Optional[str] = "norm",
                 sync_batch_norm: bool = False,
                 ):

        # Type Safety Checks
        assert isinstance(model, Module), f"Expected 'Module', got {type(model).__name__}"
        assert isinstance(data, DataModule), f"Expected 'DataModule', got {type(data).__name__}"

        self.model = model
        self.data = data

        # --- Accelerator Initialization ---
        self.project_dir = project_dir
        self.grad_accum_steps = grad_accum_steps
        self.mixed_precision = mixed_precision
        self.project_name = project_name
        self.run_config = run_config or {}
        self.log_config = log_config or {}
        self.log_with = log_with

        accelerator_kwargs = {
            "project_dir": self.project_dir,
            "log_with": self.log_with,
            "gradient_accumulation_steps": self.grad_accum_steps,
            "mixed_precision": self.mixed_precision
        }
        if accelerator_config:
            accelerator_kwargs.update(accelerator_config)

        self.accelerator = Accelerator(**accelerator_kwargs)

        # --- Reproducibility ---
        self.seed = seed
        self.deterministic = deterministic
        set_seed(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)

        self.cudnn_benchmark = cudnn_benchmark
        if self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        self.use_tf32 = use_tf32
        if self.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.do_train = do_train
        self.do_val = do_val
        self.do_test = do_test

        # --- Data Configuration ---
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataloader_config = dataloader_config or {}

        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        # --- Optimization Config ---
        self.grad_clip_value = grad_clip_value

        # Safe handling of gradient clip algorithm
        if grad_clip_algorithm:
            self.grad_clip_algorithm = grad_clip_algorithm.lower()
        else:
            self.grad_clip_algorithm = "norm"

        self.sync_batch_norm = sync_batch_norm

        self.optimizer_config = optimizer_config or {}
        self.optimizers = []
        self.schedulers = []

        # --- Compilation ---
        self.compile = compile
        self.compile_config = compile_config or {}

        # Boot
        self.setup()

    def setup(self):
        self.setup_model()
        self.setup_data()
        self.setup_accelerator()

    # =========================================================================
    # 1. MODEL SETUP
    # =========================================================================
    def setup_model(self):
        # 1. Structural Changes (SyncBN)
        # Must be done first as it modifies layer types
        if self.do_train and self.sync_batch_norm and self.accelerator.num_processes > 1:
            logger.info("Converting model to SyncBatchNorm...")
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # 2. Optimizers
        # We initialize optimizers BEFORE compilation.
        # This ensures we can access 'configure_optimizers' on the clean Module object.
        if self.do_train:
            self.setup_optimizers(**self.optimizer_config)

        # 3. Compilation
        # We compile last. torch.compile wraps the model execution logic.
        if self.compile:
            logger.info("Compiling model via torch.compile...")
            self.model = torch.compile(self.model, **self.compile_config)

    def setup_optimizers(self):
        configure_optim = getattr(self.model, "configure_optimizers", None)
        if not configure_optim:
            return

        opt_conf = configure_optim()
        if opt_conf is None:
            return

        raw_optimizers = []
        raw_schedulers = []
        common_config = {}

        # 1. PARSE CONFIG
        if isinstance(opt_conf, dict):
            common_config = opt_conf.copy()
            opt_input = common_config.pop('optimizer', None) or common_config.pop('optimizers', None)
            if opt_input is None:
                raise ValueError("Optimizer config dict must contain 'optimizer' or 'optimizers' key.")

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

        # 2. VALIDATE & REGISTER
        self.optimizers = []
        self.schedulers = []

        for i, opt in enumerate(raw_optimizers):
            if not isinstance(opt, torch.optim.Optimizer):
                raise TypeError(f"Expected torch.optim.Optimizer at index {i}, got {type(opt).__name__}")
            self.optimizers.append(opt)

        for sched_item in raw_schedulers:
            if sched_item is None:
                continue

            final_conf = common_config.copy()
            if isinstance(sched_item, dict):
                final_conf.update(sched_item)
            else:
                final_conf['scheduler'] = sched_item

            self.schedulers.append(self._standardize_scheduler(final_conf))

        if self.do_train and not self.optimizers:
            raise ValueError("Training enabled (do_train=True) but no optimizers found.")

    def _standardize_scheduler(self, sched_item) -> Dict[str, Any]:
        standard_config = {
            'scheduler': None,
            'strategy': 'epoch',
            'interval': 1,
            'monitor': None,
            'strict': True,
        }
        if isinstance(sched_item, dict):
            if 'scheduler' not in sched_item:
                raise ValueError("Scheduler dictionary must contain 'scheduler' key")
            standard_config.update(sched_item)
        else:
            standard_config['scheduler'] = sched_item
        return standard_config

    # =========================================================================
    # 2. DATA SETUP
    # =========================================================================
    def setup_data(self):
        if not (self.do_train or self.do_val or self.do_test):
            return

        if hasattr(self.data, "prepare_data") and self.accelerator.is_main_process:
            self.data.prepare_data()

        self.accelerator.wait_for_everyone()

        if hasattr(self.data, "setup"):
            sig = inspect.signature(self.data.setup)
            if 'stage' in sig.parameters:
                self.data.setup(stage="train")
            else:
                self.data.setup()

        self.setup_dataloaders()

    def setup_dataloaders(self):
        def get_loader(method, config):
            return self._fetch_loader(method, config)

        if self.do_train:
            dataset = getattr(self.data, "train_dataset", None)
            config = self._resolve_loader_config("train", dataset)
            self.train_dataloader = get_loader(getattr(self.data, "train_dataloader", None), config)
            if self.train_dataloader is None:
                logger.warning("do_train=True but no 'train_dataloader' found. Disabling training.")
                self.do_train = False

        if self.do_val:
            dataset = getattr(self.data, "val_dataset", None)
            config = self._resolve_loader_config("val", dataset)
            self.val_dataloader = get_loader(getattr(self.data, "val_dataloader", None), config)
            if self.val_dataloader is None:
                logger.warning("do_val=True but no 'val_dataloader' found. Disabling validation.")
                self.do_val = False

        if self.do_test:
            dataset = getattr(self.data, "test_dataset", None)
            config = self._resolve_loader_config("test", dataset)
            self.test_dataloader = get_loader(getattr(self.data, "test_dataloader", None), config)
            if self.test_dataloader is None:
                logger.warning("do_test=True but no 'test_dataloader' found. Disabling testing.")
                self.do_test = False

    def _resolve_loader_config(self, stage: str, dataset=None) -> Dict[str, Any]:
        """
        Merges defaults, global config, and stage-specific config.
        Automatically sets shuffle=True for Map-style training datasets.
        """
        config = {
            'num_workers': 0,
            'pin_memory': True if torch.cuda.is_available() else False,
            'persistent_workers': False,
            'prefetch_factor': None,
            'worker_init_fn': DistWorkerInitFunc(self.seed, self.accelerator.process_index),
            'shuffle': False,  # Base default
        }

        if stage == 'train':
            # MAGIC: Auto-enable shuffle for non-iterable datasets (Map-style).
            # If the user provides a standard dataset, we assume they want shuffling.
            is_iterable = isinstance(dataset, IterableDataset) if dataset else False
            if dataset is not None and not is_iterable:
                config['shuffle'] = True

            config["batch_size"] = self.train_batch_size
            config['drop_last'] = True
        else:
            config['batch_size'] = self.eval_batch_size
            config['drop_last'] = False

        # Apply global overrides (e.g., {'num_workers': 4})
        global_overrides = {
            k: v for k, v in self.dataloader_config.items()
            if not isinstance(v, dict) and not k.startswith(("train_", "val_", "test_"))
        }
        config.update(global_overrides)

        # Apply stage-specific prefixes (e.g., {'train_shuffle': True})
        prefix = f"{stage}_"
        prefix_overrides = {
            k[len(prefix):]: v for k, v in self.dataloader_config.items()
            if k.startswith(prefix)
        }
        config.update(prefix_overrides)

        # Apply nested dictionary overrides (e.g., {'train': {'shuffle': True}})
        if stage in self.dataloader_config and isinstance(self.dataloader_config[stage], dict):
            config.update(self.dataloader_config[stage])

        if config.get('persistent_workers') and config.get('num_workers', 0) == 0:
            config['persistent_workers'] = False

        self._validate_loader_config(config)
        return config

    def _validate_loader_config(self, config: Dict[str, Any]):
        valid_args = set(inspect.signature(torch.utils.data.DataLoader).parameters.keys())
        invalid_keys = [k for k in config.keys() if k not in valid_args]
        if invalid_keys:
            raise ValueError(f"Invalid DataLoader keys: {invalid_keys}. Allowed: {sorted(list(valid_args))}")

    def _fetch_loader(self, loader_method, config: Dict[str, Any]):
        """
        Inspects the dataloader method signature and passes only compatible arguments.
        This allows the user to implement simple dataloaders without accepting **kwargs.
        """
        if loader_method is None:
            return None
        remaining_config = config.copy()
        sig = inspect.signature(loader_method)
        explicit_args = {}
        has_kwargs = False

        for name, param in sig.parameters.items():
            if param.kind == param.VAR_KEYWORD:
                has_kwargs = True
                continue
            if name in remaining_config:
                explicit_args[name] = remaining_config.pop(name)

        if not has_kwargs:
            if remaining_config:
                logger.debug(f"Ignored config for {loader_method.__name__}: {list(remaining_config.keys())}")
            return loader_method(**explicit_args)

        return loader_method(**explicit_args, **remaining_config)

    # =========================================================================
    # 3. ACCELERATOR SETUP
    # =========================================================================

    def setup_accelerator(self):
        if self.log_with and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.project_name,
                config=self.run_config,
                init_kwargs=self.log_config
            )

        if self.do_train:
            raw_schedulers = [s['scheduler'] for s in self.schedulers]
            to_prepare = [self.model]
            if self.optimizers:
                to_prepare.extend(self.optimizers)
            if raw_schedulers:
                to_prepare.extend(raw_schedulers)

            prepared_objs = self.accelerator.prepare(*to_prepare)

            self.model = prepared_objs[0]
            curr = 1
            if self.optimizers:
                self.optimizers = list(prepared_objs[curr : curr + len(self.optimizers)])
                curr += len(self.optimizers)
            if raw_schedulers:
                prepared_schedulers = prepared_objs[curr:]
                for i, prepared_scheduler in enumerate(prepared_schedulers):
                    self.schedulers[i]['scheduler'] = prepared_scheduler
        else:
            self.model = self.accelerator.prepare(self.model)

        if self.train_dataloader:
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
        if self.val_dataloader:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        if self.test_dataloader:
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

    # =========================================================================
    # 4. HELPER FUNCTIONS
    # =========================================================================

    @property
    def raw_model(self):
        """Returns the underlying model, stripped of DDP/FSDP wrappers."""
        return self.accelerator.unwrap_model(self.model)

    @property
    def device(self):
        """Current device (GPU/CPU)."""
        return self.accelerator.device

    @property
    def autocast(self):
        """Returns the accelerator's autocast context manager."""
        return self.accelerator.autocast()

    @property
    def is_main_process(self):
        """Returns True if this is the main process (Rank 0)."""
        return self.accelerator.is_main_process

    @property
    def num_processes(self):
        return self.accelerator.num_processes

    @property
    def sync_gradients(self):
        """Returns True if gradients will be synced this iteration."""
        return self.accelerator.sync_gradients

    def free_memory(self):
        """Aggressively reclaims GPU memory."""
        self.optimizers_zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def wait(self):
        """Barrier."""
        self.accelerator.wait_for_everyone()

    def print(self, *args, **kwargs):
        """Safe print function."""
        self.accelerator.print(*args, **kwargs)

    def get_logger(self, name: str = "wandb"):
        """Returns the underlying tracker object."""
        return self.accelerator.get_tracker(name)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Logs metrics using the trackers."""
        self.accelerator.log(metrics, step=step)

    def finish(self):
        """Closes all loggers."""
        self.accelerator.end_training()

    # =========================================================================
    # 5. DISTRIBUTED DATA HELPERS
    # =========================================================================

    def gather(self, tensor: torch.Tensor, exact: bool = False) -> torch.Tensor:
        """Gathers a tensor from all processes."""
        if exact:
            return self.accelerator.gather_for_metrics(tensor)
        return self.accelerator.gather(tensor)

    def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Reduces a tensor across all processes."""
        return self.accelerator.reduce(tensor, reduction=reduction)

    # =========================================================================
    # 6. OPTIMIZATION & SCHEDULING
    # =========================================================================

    def backward(self, loss: torch.Tensor, **kwargs):
        """Calculates gradients."""
        self.accelerator.backward(loss, **kwargs)

    def accumulate(self):
        """Context manager for gradient accumulation."""
        return self.accelerator.accumulate(self.model)

    def get_lr(self, optimizer=None) -> float:
        """Returns the learning rate of the first param group."""
        opt = optimizer or self.optimizers[0]
        return opt.param_groups[0]['lr']

    def optimizer_step(self, optimizer, zero_grad: bool = False, set_to_none: bool = True, **kwargs):
        """
        Steps the optimizer.

        Args:
            optimizer: The optimizer to step.
            zero_grad (bool): If True, zeros gradients AFTER the step.
            set_to_none (bool): Used if zero_grad is True.
            **kwargs: Passed directly to optimizer.step() (e.g. 'closure').
        """
        # Explicitly separate zero_grad logic from step logic.
        optimizer.step(**kwargs)
        if zero_grad:
            self.optimizer_zero_grad(optimizer, set_to_none=set_to_none)

    def optimizer_zero_grad(self, optimizer, set_to_none: bool = True):
        """Zeroes the gradients of a specific optimizer."""
        optimizer.zero_grad(set_to_none=set_to_none)

    def optimizers_step(self, zero_grad: bool = False, set_to_none: bool = True, **kwargs):
        """Steps all registered optimizers."""
        for opt in self.optimizers:
            self.optimizer_step(opt, zero_grad=zero_grad, set_to_none=set_to_none, **kwargs)

    def optimizers_zero_grad(self, set_to_none: bool = True):
        """Zeroes gradients for all registered optimizers."""
        for opt in self.optimizers:
            self.optimizer_zero_grad(opt, set_to_none=set_to_none)

    def schedulers_step(self, counter: int, strategy: str = "step", monitor: Optional[Dict[str, Any]] = None):
        """Updates schedulers based on strategy and interval."""
        for scheduler in self.schedulers:
            self.scheduler_step(scheduler, counter, strategy, monitor)

    def scheduler_step(self, scheduler: Dict[str, Any], counter: int, strategy: str = "step", monitor: Optional[Dict[str, Any]] = None):
        monitor_key = scheduler['monitor']
        raw_scheduler = scheduler['scheduler']

        if scheduler['strategy'] == strategy and counter % scheduler['interval'] == 0:
            if monitor_key:
                if monitor and monitor_key in monitor:
                    metric = monitor[monitor_key]
                    raw_scheduler.step(metric)
                else:
                    if scheduler['strict']:
                        raise ValueError(f"Scheduler requires metric '{monitor_key}' but it was not found in self.monitor.")
                    else:
                        logger.warning(f"Scheduler metric '{monitor_key}' missing. Skipping step.")
            else:
                raw_scheduler.step()

    def gradient_clip_step(self) -> Optional[torch.Tensor]:
        """Performs gradient clipping."""
        if self.grad_clip_value is None or self.grad_clip_value <= 0:
            return None

        grad_norm = None
        if self.grad_clip_algorithm == "norm":
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_value
            )
        elif self.grad_clip_algorithm == "value":
            self.accelerator.clip_grad_value_(
                self.model.parameters(), self.grad_clip_value
            )
        return grad_norm

    # =========================================================================
    # 7. MODEL PERSISTENCE
    # =========================================================================

    def save_state(self, path: str, **kwargs):
        """Saves the FULL training state via Accelerator."""
        self.wait()
        self.accelerator.save_state(path, **kwargs)
        self.print(f"Training state checkpoint saved to: {path}")

    def load_state(self, path: str, **kwargs):
        """Restores FULL training state."""
        self.print(f"Loading training state from: {path}")
        self.accelerator.load_state(path, **kwargs)

    def save_model(self, target_dir: str, **kwargs):
        """Saves the model weights (unwrapped) to the directory."""
        self.wait()
        self.accelerator.save_model(self.model, target_dir, **kwargs)
        self.print(f"Model weights saved to: {target_dir}")

    def load_model(self, path: str, strict: bool = True, **kwargs):
        """Smart loading of model weights to the current device."""
        # 1. Resolve Path
        file_path = path
        if os.path.isdir(path):
            st_path = os.path.join(path, "model.safetensors")
            bin_path = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(st_path): file_path = st_path
            elif os.path.exists(bin_path): file_path = bin_path

        self.print(f"Loading model weights from: {file_path}")

        # 2. Set Default Map Location to Runner's Device
        if "map_location" not in kwargs:
            kwargs["map_location"] = self.device

        # 3. Load
        if file_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
                # Safetensors only accepts 'device', not 'map_location' or other loader kwargs.
                device = kwargs.get("device", self.device)

                # We strictly pass only the device to avoid crashes with incompatible kwargs.
                state_dict = load_file(file_path, device=device)
            except ImportError:
                raise ImportError("Found .safetensors file but 'safetensors' library is not installed.")
        else:
            # torch.load requires map_location, not device.
            if "device" in kwargs:
                kwargs.pop("device")
            state_dict = torch.load(file_path, **kwargs)

        # 4. Apply
        self.raw_model.load_state_dict(state_dict, strict=strict)
