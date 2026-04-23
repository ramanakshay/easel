import os
import inspect
import logging
import gc
from typing import Any, Dict, List, Optional, Union

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, TorchDynamoPlugin

from ..data import Data
from ..model import Model

logger = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(self,
                 # 1. Core
                 data: Data,
                 model: Model,
                 do_train: bool = False,
                 do_val: bool = False,
                 do_test: bool = False,
                 do_predict: bool = False,

                 # 2. Data
                 stage: Optional[str] = None,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 dataloader_config: Optional[Dict[str, Any]] = None,

                 # 3. Hardware & Optimization
                 mixed_precision: str = "no",
                 gradient_accumulation_steps: int = 1,
                 compile: bool = False,
                 gradient_clip_value: float = 0.0,
                 gradient_clip_algorithm: str = "norm",
                 sync_batch_norm: bool = False,
                 optimizer_config: Optional[Dict[str, Any]] = None,
                 accelerator_config: Optional[Dict[str, Any]] = None,

                 # 4. Global Configs
                 seed: int = 42,
                 deterministic: bool = False,
                 tf32: Union[bool, str] = False,
                 cudnn_benchmark: bool = False,

                 # 5. Logging
                 project_dir: str = "",
                 project_name: str = "",
                 run_config: Optional[Dict[str, Any]] = None,
                 log_with: Union[str, List[str], None] = None,
                 log_config: Optional[Dict[str, Any]] = None,
                 ):

         # --- 1. Model ---
        self.model = model
        self.optimizer_config = optimizer_config or {}
        self.optimizers = []
        self.schedulers = []
        self.monitor = {}


        # --- 2. Data ---
        self.data = data
        self.do_train = do_train
        self.do_val = do_val
        self.do_test = do_test
        self.do_predict = do_predict
        self.stage = stage
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataloader_config = dataloader_config or {}
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.predict_dataloader = None

        # --- 3. Hardware & Optimization Assignment ---
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.compile = compile
        self.gradient_clip_value = gradient_clip_value
        self.gradient_clip_algorithm = gradient_clip_algorithm.lower()
        self.sync_batch_norm = sync_batch_norm
        self.accelerator_config = accelerator_config or {}


        # --- 4. Global Configs Assignment ---
        self.seed = seed
        self.deterministic = deterministic
        self.tf32 = tf32
        self.cudnn_benchmark = cudnn_benchmark

        # --- 5. Logging Assignment ---
        self.project_dir = project_dir
        self.project_name = project_name
        self.run_config = run_config or {}
        self.log_with = log_with
        self.log_config = log_config or {}

        # --- Boot Sequence ---
        self.setup_accelerator()
        self.setup_globals()
        self.setup_data()
        self.setup_model()

    def setup_globals(self):
        """Step 1: Sets up PyTorch global hardware and reproducibility flags."""

        # 1. Seed across all devices (Python, Numpy, PyTorch CPU/GPU)
        set_seed(self.seed, device_specific=True)

        # 2. Determinism
        if self.deterministic:
            if self.cudnn_benchmark:
                logger.warning("cudnn_benchmark cannot be True if deterministic is True. Disabling benchmark.")
                self.cudnn_benchmark = False

            # PyTorch 2.0+ REQUIRES this environment variable for deterministic CUDA ops.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True)

        # 3. cuDNN Benchmark
        if self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # 4. TF32 (TensorFloat-32)
        if self.tf32:
            precision = self.tf32 if isinstance(self.tf32, str) else "high"
            torch.set_float32_matmul_precision(precision)


    def setup_accelerator(self):
        """Step 2: Builds the Accelerator object with plugins and trackers."""

        # Construct the core arguments for the Accelerator
        accelerator_kwargs = {
            "project_dir": self.project_dir,
            "log_with": self.log_with,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
        }

        # Apply any advanced overrides from the escape hatch
        accelerator_kwargs.update(self.accelerator_config)

        # Handle TorchDynamo (torch.compile) via Accelerate Plugin safely
        if self.compile and "dynamo_plugin" not in accelerator_kwargs:
            dynamo_plugin = TorchDynamoPlugin(
                backend="inductor",
                mode="default",
                fullgraph=False,
                dynamic=False
            )
            accelerator_kwargs["dynamo_plugin"] = dynamo_plugin

        # Initialize the Accelerator
        self.accelerator = Accelerator(**accelerator_kwargs)

        # Initialize Trackers (W&B, TensorBoard, MLflow, etc.)
        if self.log_with and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.project_name,
                config=self.run_config,
                init_kwargs=self.log_config
            )

    def setup_data(self):
        """Step 3: Prepares datasets and builds distributed DataLoaders."""
        if self.data is None:
            return

        # 1. Prepare Data (Main Process Only)
        if hasattr(self.data, "prepare"):
            if self.accelerator.is_main_process:
                self.data.prepare()
            # Force all GPUs to wait until rank 0 finishes downloading/tokenizing
            self.accelerator.wait_for_everyone()

        # 2. Setup Data (Pass 'stage' if requested)
        if hasattr(self.data, "setup"):
            sig = inspect.signature(self.data.setup)
            if 'stage' in sig.parameters:
                self.data.setup(stage=self.stage)
            else:
                self.data.setup()

        # 3. Fetch DataLoaders dynamically
        loaders_to_prepare = []
        modes = []

        for mode in ["train", "val", "test", "predict"]:
            # Only fetch if the flag (do_train, do_val) is True
            if getattr(self, f"do_{mode}"):
                kwargs = self._get_dataloader_kwargs(mode)
                loader = self._fetch_loader(mode, kwargs)

                # Store the raw loader
                setattr(self, f"{mode}_dataloader", loader)

                if loader is not None:
                    loaders_to_prepare.append(loader)
                    modes.append(mode)

        # 4. Prepare DataLoaders with Accelerate
        if loaders_to_prepare:
            prepared_loaders = self.accelerator.prepare(*loaders_to_prepare)

            # Safe unpacking: Accelerator returns a single object if only 1 loader is passed
            if not isinstance(prepared_loaders, tuple):
                prepared_loaders = (prepared_loaders,)

            # Re-assign the wrapped loaders back to the runner variables
            for i, mode in enumerate(modes):
                setattr(self, f"{mode}_dataloader", prepared_loaders[i])


    def _get_dataloader_kwargs(self, mode: str) -> Dict[str, Any]:
        """
        Resolves the hierarchical dataloader config for a specific mode.
        Hierarchy: Exposed defaults -> Global config -> Mode prefix config -> Mode section config.
        """
        # 1. Base Exposed Arguments
        kwargs = {
            "batch_size": self.train_batch_size if mode == "train" else self.eval_batch_size
        }

        if not self.dataloader_config:
            return kwargs

        # 2. Global Arguments (e.g., {"pin_memory": True})
        # Ignore dictionaries (sections) and prefixed keys (train_, val_)
        mode_prefixes = ("train_", "val_", "test_", "predict_")
        for k, v in self.dataloader_config.items():
            if not isinstance(v, dict) and not k.startswith(mode_prefixes):
                kwargs[k] = v

        # 3. Prefixed Arguments (e.g., {"train_drop_last": True})
        prefix = f"{mode}_"
        for k, v in self.dataloader_config.items():
            if k.startswith(prefix):
                kwargs[k[len(prefix):]] = v

        # 4. Section Arguments (e.g., {"predict": {"batch_size": 10}})
        section = self.dataloader_config.get(mode)
        if isinstance(section, dict):
            kwargs.update(section)

        return kwargs


    def _fetch_loader(self, mode: str, kwargs: Dict[str, Any]) -> Any:
        """
        Inspects the user's dataloader method signature and safely passes the resolved kwargs.
        """
        method_name = f"{mode}_dataloader"
        method = getattr(self.data, method_name, None)

        if not method:
            return None

        sig = inspect.signature(method)

        # Check if the method accepts arbitrary **kwargs
        accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

        if accepts_kwargs:
            # Pass everything safely
            return method(**kwargs)
        else:
            # Filter kwargs to ONLY those explicitly requested by the user's signature
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

            # Warn the user if we are silently dropping configs they requested
            dropped_keys = set(kwargs.keys()) - set(valid_kwargs.keys())
            if dropped_keys:
                logger.debug(
                    f"Ignored kwargs for `{method_name}` because they are not in the signature: {dropped_keys}. "
                    f"To use them, add `**kwargs` to your method definition."
                )

            return method(**valid_kwargs)

    def setup_model(self):
        """Step 4: Prepares the model, optimizers, and schedulers for training."""

        # 1. Fast path for inference/testing
        if not self.do_train:
            logger.info("do_train=False. Skipping optimizers and preparing model for inference.")
            prepared_objs = self.accelerator.prepare(self.model)
            if not isinstance(prepared_objs, tuple):
                prepared_objs = (prepared_objs,)
            self.model = prepared_objs[0]
            return

        # 2. Structural Changes (SyncBatchNorm)
        # Must be done BEFORE optimizers are created so they track the correct parameters
        if self.sync_batch_norm and self.accelerator.num_processes > 1:
            logger.info("Converting model to SyncBatchNorm...")
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # 3. Dynamically call configure_optimizers
        configure_optim = getattr(self.model, "configure_optimizers", None)
        if not configure_optim:
            raise NotImplementedError("To train, you must implement `configure_optimizers()` in your Module.")

        # Safely inject optimizer_config based on signature
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

        # 4. Parse and Standardize Output
        self._standardize_optimizers(opt_conf)

        # 5. Prepare Everything with Accelerate
        # We must prepare the model, optimizers, and schedulers simultaneously
        to_prepare = [self.model] + self.optimizers + [s['scheduler'] for s in self.schedulers]

        prepared_objs = self.accelerator.prepare(*to_prepare)

        # Safe unpacking
        if not isinstance(prepared_objs, tuple):
            prepared_objs = (prepared_objs,)

        # Re-assign Model
        self.model = prepared_objs[0]

        # Re-assign Optimizers
        curr_idx = 1
        if self.optimizers:
            self.optimizers = list(prepared_objs[curr_idx : curr_idx + len(self.optimizers)])
            curr_idx += len(self.optimizers)

        # Re-assign Schedulers
        if self.schedulers:
            prepared_schedulers = prepared_objs[curr_idx:]
            for i, prep_sched in enumerate(prepared_schedulers):
                self.schedulers[i]['scheduler'] = prep_sched


    def _standardize_optimizers(self, opt_conf: Any):
        """
        Parses the output of configure_optimizers into strict lists of
        optimizers and standardized scheduler dictionaries.
        """
        raw_optimizers = []
        raw_schedulers = []
        common_config = {}

        # Parse Dictionary (e.g. {'optimizer': Adam, 'lr_scheduler': {'scheduler': StepLR, 'interval': 'step'}})
        if isinstance(opt_conf, dict):
            common_config = opt_conf.copy()
            opt_input = common_config.pop('optimizer', None) or common_config.pop('optimizers', None)
            if opt_input is None:
                raise ValueError("Optimizer config dict must contain an 'optimizer' or 'optimizers' key.")

            raw_optimizers = opt_input if isinstance(opt_input, list) else [opt_input]

            sched_input = common_config.pop('scheduler', None) or common_config.pop('schedulers', None)
            if sched_input is not None:
                raw_schedulers = sched_input if isinstance(sched_input, list) else [sched_input]

        # Parse Tuple (e.g. (Adam, StepLR) or ([Adam], [StepLR]))
        elif isinstance(opt_conf, tuple):
            if len(opt_conf) != 2:
                raise ValueError(f"Tuple config must be (Optimizers, Schedulers). Got length {len(opt_conf)}")
            raw_optimizers = opt_conf[0] if isinstance(opt_conf[0], list) else [opt_conf[0]]
            raw_schedulers = opt_conf[1] if isinstance(opt_conf[1], list) else [opt_conf[1]]

        # Parse Single Optimizer or List of Optimizers
        else:
            raw_optimizers = opt_conf if isinstance(opt_conf, list) else [opt_conf]

        # 1. Register Optimizers
        for i, opt in enumerate(raw_optimizers):
            if not isinstance(opt, torch.optim.Optimizer):
                raise TypeError(f"Expected torch.optim.Optimizer at index {i}, got {type(opt).__name__}")
            self.optimizers.append(opt)

        # 2. Register & Standardize Schedulers
        for sched_item in raw_schedulers:
            if sched_item is None:
                continue

            # The default standard dictionary structure
            std_sched = {
                'scheduler': None,
                'strategy': 'epoch',  # "epoch" or "step"
                'interval': 1,
                'monitor': None,      # Key string used for ReduceLROnPlateau
                'strict': True,       # Crash if monitor key is missing
            }

            if isinstance(sched_item, dict):
                if 'scheduler' not in sched_item:
                    raise ValueError("Scheduler dictionary must contain a 'scheduler' key")
                std_sched.update(common_config) # Merge global dict configs
                std_sched.update(sched_item)    # Override with specific scheduler configs
            else:
                std_sched.update(common_config)
                std_sched['scheduler'] = sched_item

            self.schedulers.append(std_sched)

    # =========================================================================
    # 1. Properties (State & Hardware)
    # =========================================================================
    @property
    def raw_model(self) -> torch.nn.Module:
        """Returns the underlying model, stripping away DDP/Accelerate wrappers."""
        return self.accelerator.unwrap_model(self.model)

    @property
    def device(self) -> torch.device:
        """Returns the current hardware device for this process."""
        return self.accelerator.device

    @property
    def is_main_process(self) -> bool:
        """True only on rank 0. Useful for logging and saving."""
        return self.accelerator.is_main_process

    @property
    def num_processes(self) -> int:
        """Total number of GPUs/TPUs participating in training."""
        return self.accelerator.num_processes

    @property
    def sync_gradients(self) -> bool:
        """True if gradients will be synchronized this step (handles accumulation logic)."""
        return self.accelerator.sync_gradients

    # =========================================================================
    # 2. Context Managers
    # =========================================================================
    def autocast(self):
        """Context manager for executing operations in mixed precision."""
        return self.accelerator.autocast()

    def accumulate(self):
        """Context manager for gradient accumulation. Handles `no_sync` for DDP."""
        return self.accelerator.accumulate(self.model)

    # =========================================================================
    # 3. Distributed Operations & Logging
    # =========================================================================
    def wait(self):
        """Blocks all processes until everyone reaches this point."""
        self.accelerator.wait_for_everyone()

    def print(self, *args, **kwargs):
        """Safe print function that ONLY prints on the main process."""
        self.accelerator.print(*args, **kwargs)

    def log(self, values: Dict[str, Any], step: Optional[int] = None):
        """Logs metrics to the active trackers (W&B, TensorBoard)."""
        self.accelerator.log(values, step=step)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gathers a tensor from all GPUs to all GPUs. Useful for contrastive loss."""
        return self.accelerator.gather(tensor)

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.accelerator.gather_for_metrics(tensor)

    def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Reduces a tensor across all GPUs (supports 'mean', 'sum', 'min', 'max')."""
        return self.accelerator.reduce(tensor, reduction=reduction)


    # =========================================================================
    # 4. Optimization Engine
    # =========================================================================

    def backward(self, loss: torch.Tensor, **kwargs):
        """Scales the loss and computes gradients."""
        self.accelerator.backward(loss, **kwargs)

    def clip_gradients(self):
        """Clips gradients across all processes based on init configurations."""
        if self.sync_gradients and self.gradient_clip_value > 0.0:
            if self.gradient_clip_algorithm == "value":
                self.accelerator.clip_grad_value_(self.model.parameters(), self.gradient_clip_value)
            else:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)

    # --- Zero Grad ---
    def optimizer_zero_grad(self, optimizer: torch.optim.Optimizer, set_to_none: bool = True):
        """Zeroes gradients for a specific optimizer object."""
        optimizer.zero_grad(set_to_none=set_to_none)

    def optimizers_zero_grad(self, set_to_none: bool = True):
        """Zeroes gradients for all registered optimizers."""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    # --- Optimizer Step ---
    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        """Steps a specific optimizer object."""
        optimizer.step()

    def optimizers_step(self):
        """Steps all registered optimizers."""
        for opt in self.optimizers:
            opt.step()

    # --- Scheduler Step ---
    def _execute_scheduler_step(self, sched_dict: Dict[str, Any]):
        """Internal helper to safely step a scheduler, checking the monitor dictionary."""
        scheduler = sched_dict['scheduler']
        monitor_key = sched_dict['monitor']

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if monitor_key not in self.monitor:
                if sched_dict['strict']:
                    raise KeyError(
                        f"Scheduler expected '{monitor_key}' in `runner.monitor`, but it was not found. "
                        f"Set strict=False to ignore, or populate `runner.monitor['{monitor_key}']` before stepping."
                    )
                else:
                    logger.debug(f"Skipping scheduler step: '{monitor_key}' missing from monitor.")
                    return

            # Step with the tracked metric
            scheduler.step(self.monitor[monitor_key])
        else:
            scheduler.step()

    def scheduler_step(self, scheduler: Any, strategy: str, counter: int):
        """
        Steps a specific scheduler object if it matches the strategy and interval.
        """
        # Find the configuration dictionary for this specific scheduler object
        for sched_dict in self.schedulers:
            if sched_dict['scheduler'] is scheduler:
                if sched_dict['strategy'] == strategy and counter % sched_dict['interval'] == 0:
                    self._execute_scheduler_step(sched_dict)
                return

        logger.warning("Scheduler object not found in runner.schedulers.")

    def schedulers_step(self, strategy: str, counter: int):
        """
        Steps all registered schedulers whose strategy matches the current mode (e.g., "epoch" or "step")
        AND whose interval aligns with the current counter.
        """
        for sched_dict in self.schedulers:
            if sched_dict['strategy'] == strategy:
                if counter % sched_dict['interval'] == 0:
                    self._execute_scheduler_step(sched_dict)

    # =========================================================================
    # 5. State Management
    # =========================================================================
    def save_state(self, output_dir: Optional[str] = None):
        """Saves model, optimizers, and RNG states securely across all processes."""
        self.wait()
        save_path = output_dir or self.project_dir
        if self.is_main_process:
            logger.info(f"Saving runner state to {save_path}...")
        self.accelerator.save_state(save_path)

    def load_state(self, input_dir: Optional[str] = None):
        """Restores model, optimizers, and RNG states."""
        self.wait()
        load_path = input_dir or self.project_dir
        if self.is_main_process:
            logger.info(f"Loading runner state from {load_path}...")
        self.accelerator.load_state(load_path)

    def free_memory(self):
        """
        Forces garbage collection and empties the CUDA cache.
        WARNING: Causes severe fragmentation if called inside the step loop.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
