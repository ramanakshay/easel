"""Easel Supervised Learning Engine.

This module defines the :class:`Engine` class, which orchestrates the full
machine learning lifecycle: training, validation, testing, and prediction.

Public objects:
    Engine: Orchestrates the training, validation, testing, and prediction
        loops.
"""

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
    """Orchestrates the training, validation, testing, and prediction loops.

    Subclass :class:`Engine`, implement the ``*_step`` methods
    (:meth:`train_step`, :meth:`val_step`, :meth:`test_step`,
    :meth:`predict_step`), and optionally override any of the lifecycle
    hooks (``on_*``). Then construct the engine and call :meth:`run`.

    The engine wraps HuggingFace ``accelerate`` for device placement,
    distributed training, mixed precision, gradient accumulation, and
    experiment tracking. The :class:`~easel.data.Data` object provides
    dataloaders and the :class:`~easel.model.Model` provides the network
    and optimizer configuration.

    Key attributes:
        model: The (possibly wrapped) :class:`~torch.nn.Module` being trained.
        data: The :class:`~easel.data.Data` instance providing dataloaders.
        accelerator: The underlying ``accelerate.Accerator`` instance.
        optimizers: List of prepared optimizers.
        schedulers: List of scheduler config dicts with keys ``scheduler``,
            ``strategy``, ``interval``, ``monitor``.
        monitor: Dict populated by the user for schedulers
            that need a monitored metric value.
        step: Global optimizer-step count (incremented after each gradient
            sync boundary).
        epoch: Current epoch index (incremented at epoch end).
        should_stop: Set to ``True`` from a hook to early-stop training.
        train_dataloader: Prepared training dataloader (or ``None``).
        val_dataloader: Prepared validation dataloader (or ``None``).
        test_dataloader: Prepared test dataloader (or ``None``).
        predict_dataloader: Prepared prediction dataloader (or ``None``).
    """

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
                 val_steps: Optional[int] = None,
                 test_steps: Optional[int] = None,
                 predict_steps: Optional[int] = None,

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
        """Initialize the engine and run all setup phases.

        Args:
            data: The :class:`~easel.data.Data` instance providing dataloaders.
            model: The :class:`~easel.model.Model` instance to train or evaluate.
            do_train: Whether to run the training loop.
            do_val: Whether to run the validation loop.
            do_test: Whether to run the testing loop.
            do_predict: Whether to run the prediction loop.
            max_epochs: Maximum number of epochs to train. Required if
                ``max_steps`` is ``None`` and ``do_train`` is ``True``.
            max_steps: Maximum number of optimizer steps to train. If both
                ``max_epochs`` and ``max_steps`` are set, ``max_steps`` wins
                as the stop condition.
            train_steps_per_epoch: Number of optimizer steps per epoch.
                Auto-calculated from the dataloader length when not set.
            val_steps: Maximum number of validation batches per run.
            test_steps: Maximum number of test batches per run.
            predict_steps: Maximum number of prediction batches per run.
            val_strategy: When to run validation: ``"epoch"``, ``"step"``,
                or ``"no"`` (validation disabled within training).
            val_start: Step or epoch index at which validation begins.
            val_interval: Run validation every ``val_interval`` steps or
                epochs (depending on ``val_strategy``).
            project_dir: Directory for accelerator outputs.
            project_name: Experiment tracker project name.
            log_with: Tracker(s) to log with (e.g. ``"wandb"``,
                ``"tensorboard"``).
            init_trackers_config: Extra config for ``accelerator.init_trackers``.
            stage: Stage string forwarded to :meth:`Data.setup`.
            train_batch_size: Default training batch size.
            eval_batch_size: Default evaluation batch size.
            dataloader_config: Per-mode or global dataloader kwargs. See
                :meth:`_get_dataloader_kwargs` for the supported styles.
            optimizers_config: Kwargs forwarded to
                :meth:`Model.configure_optimizers`.
            gradient_accumulation_steps: Number of micro-batches accumulated
                per optimizer step.
            gradient_clip_value: Value for gradient clipping (or ``None``).
            gradient_clip_algorithm: ``"norm"`` or ``"value"``.
            mixed_precision: ``"no"``, ``"fp16"``, ``"bf16"``, or ``"fp8"``.
            compile: Whether to compile the model with ``torch.compile``.
            sync_batch_norm: Whether to convert to ``SyncBatchNorm`` (multi-GPU).
            accelerator_config: Extra kwargs forwarded to ``Accelerator``.
            seed: Random seed for reproducibility (or ``None``).
            deterministic: If ``True``, enable deterministic algorithms.
            tf32: If truthy, enable TF32. A string sets the matmul precision
                (``"high"`` or ``"medium"``); ``True`` uses ``"high"``.
            cudnn_benchmark: If ``True``, enable ``cudnn.benchmark``.

        Raises:
            ValueError: If both ``max_epochs`` and ``max_steps`` are ``None``
                while ``do_train=True``, or if ``max_epochs`` is set but the
                train dataloader length is unknown (e.g. an
                :class:`~torch.utils.data.IterableDataset`) and
                ``train_steps_per_epoch`` was not provided.
        """

        self.model = model
        self.data = data

        self.do_train = do_train
        self.do_val = do_val
        self.do_test = do_test
        self.do_predict = do_predict

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.train_steps_per_epoch = train_steps_per_epoch
        self.val_steps = val_steps
        self.test_steps = test_steps
        self.predict_steps = predict_steps

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

    def setup_globals(self) -> None:
        """Configure global PyTorch flags for determinism and precision.

        Sets deterministic algorithms, ``cudnn.benchmark``, and TF32 matmul
        precision according to the corresponding constructor arguments.
        Warns and disables ``cudnn_benchmark`` if both ``deterministic`` and
        ``cudnn_benchmark`` were requested.
        """
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

    def setup_accelerator(self) -> None:
        """Construct and configure the ``accelerate.Accelerator``.

        Merges named accelerator arguments with ``accelerator_config``
        (the latter takes precedence, with a warning on overlap). Optionally
        installs a TorchDynamo plugin when ``compile=True``. Seeds the
        environment and initializes experiment trackers on the main process.
        """
        named_args = {
            "project_dir": self.project_dir,
            "log_with": self.log_with,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
        }

        overlap = set(self.accelerator_config.keys()) & set(named_args.keys())
        if overlap:
            logger.warning(
                f"accelerator_config keys {overlap} overlap with named arguments. "
                f"accelerator_config values take precedence."
            )

        accelerator_kwargs = named_args.copy()
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
            tracker_kwargs = self.init_trackers_config.copy()
            project_name = tracker_kwargs.pop("project_name")
            config = tracker_kwargs.pop("config", None)
            init_kwargs = tracker_kwargs.pop("init_kwargs", None)
            self.accelerator.init_trackers(project_name, config=config, init_kwargs=init_kwargs)

    # ------------------------------------------------------------------
    # Setup: data
    # ------------------------------------------------------------------

    def setup_data(self) -> None:
        """Prepare data, build dataloaders, and derive loop limits.

        Runs :meth:`Data.prepare` on the main process, then calls
        :meth:`Data.setup`. For each enabled mode, builds the corresponding
        dataloader via :meth:`_fetch_loader`, prepares it with the
        accelerator, and disables any mode whose dataloader is ``None``.

        Auto-derives ``*_steps_per_epoch`` / ``*_steps`` from the dataloader
        length when not explicitly set (training steps are divided by
        ``gradient_accumulation_steps``). Reconciles ``max_epochs`` and
        ``max_steps`` — at least one must be set when ``do_train=True``.

        Raises:
            ValueError: If both ``max_epochs`` and ``max_steps`` are ``None``
                while ``do_train=True``, or if ``max_epochs`` is set but
                the train dataloader length is unknown (e.g. an
                :class:`~torch.utils.data.IterableDataset`) and
                ``train_steps_per_epoch`` was not provided.
        """
        if self.data is None:
            return

        if self.accelerator.is_main_process:
            self.data.prepare()
        self.accelerator.wait_for_everyone()

        self.data.setup(stage=self.stage)

        modes = ["train", "val", "test", "predict"]
        enabled_modes = []
        loaders = []

        for mode in modes:
            if not getattr(self, f"do_{mode}"):
                continue

            kwargs = self._get_dataloader_kwargs(mode)
            loader = self._fetch_loader(mode, kwargs)

            if loader is None:
                setattr(self, f"do_{mode}", False)
            else:
                setattr(self, f"{mode}_dataloader", loader)
                enabled_modes.append(mode)
                loaders.append(loader)

        if loaders:
            prepared = self.accelerator.prepare(*loaders)
            if not isinstance(prepared, tuple):
                prepared = (prepared,)
            for i, mode in enumerate(enabled_modes):
                setattr(self, f"{mode}_dataloader", prepared[i])

        for mode in modes:
            if not getattr(self, f"do_{mode}"):
                continue
            steps_attr = f"{mode}_steps_per_epoch" if mode == "train" else f"{mode}_steps"
            if getattr(self, steps_attr) is not None:
                continue
            loader = getattr(self, f"{mode}_dataloader")
            if loader is None:
                continue
            try:
                num_batches = len(loader)
                if mode == "train":
                    num_batches = math.ceil(num_batches / self.gradient_accumulation_steps)
                setattr(self, steps_attr, num_batches)
            except TypeError:
                pass

        if self.do_train:
            if self.max_epochs is None and self.max_steps is None:
                raise ValueError("At least one of max_epochs or max_steps must be specified.")
            if self.max_steps is None and self.max_epochs is not None:
                if self.train_steps_per_epoch is not None:
                    self.max_steps = self.max_epochs * self.train_steps_per_epoch
                else:
                    raise ValueError(
                        "Could not determine `train_steps_per_epoch` for the train dataloader "
                        "(likely an IterableDataset). When using an iterable dataset, set "
                        "`max_steps` explicitly, or pass `train_steps_per_epoch` so `max_epochs` "
                        "can be converted to a step limit."
                    )
            if self.max_epochs is None and self.max_steps is not None:
                if self.train_steps_per_epoch is not None:
                    self.max_epochs = math.ceil(self.max_steps / self.train_steps_per_epoch)

    def _get_dataloader_kwargs(self, mode: str) -> Dict[str, Any]:
        """Resolve dataloader kwargs for the given mode.

        Supports three styles within ``dataloader_config``:

        1. Global keys applied to all modes (e.g. ``{"shuffle": False}``).
        2. Mode-prefixed keys (e.g. ``"train_num_workers"`` →
           ``num_workers`` for the train loader only).
        3. Per-mode sub-dicts (e.g. ``{"train": {"batch_size": 8}}``).

        A ``batch_size`` default is always applied if not otherwise set:
        ``train_batch_size`` for the train mode, ``eval_batch_size``
        otherwise.

        Args:
            mode: One of ``"train"``, ``"val"``, ``"test"``, ``"predict"``.

        Returns:
            A dict of kwargs suitable for passing to the mode's
            ``*_dataloader`` method.
        """
        default_batch_size = self.train_batch_size if mode == "train" else self.eval_batch_size

        if not self.dataloader_config:
            return {"batch_size": default_batch_size}

        kwargs: Dict[str, Any] = {}

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

        kwargs.setdefault("batch_size", default_batch_size)
        return kwargs

    def _fetch_loader(self, mode: str, kwargs: Dict[str, Any]) -> Any:
        """Call the ``<mode>_dataloader`` method on ``self.data``.

        Introspects the method signature: if it accepts ``**kwargs``, all
        kwargs are forwarded. Otherwise, only parameters the method declares
        are passed, and dropped keys are logged at debug level.

        Args:
            mode: One of ``"train"``, ``"val"``, ``"test"``, ``"predict"``.
            kwargs: Kwargs resolved by :meth:`_get_dataloader_kwargs`.

        Returns:
            The dataloader returned by the method, or ``None``.
        """
        method_name = f"{mode}_dataloader"
        method = getattr(self.data, method_name)

        sig = inspect.signature(method)
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

        if accepts_kwargs:
            return method(**kwargs)

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

    def setup_model(self) -> None:
        """Prepare the model, optimizers, and schedulers for training.

        If ``do_train=False``, only prepares the model for inference and
        returns. Otherwise, optionally converts to ``SyncBatchNorm``, calls
        ``model.configure_optimizers(**optimizers_config)``, normalizes the
        return via :meth:`_standardize_optimizers`, and prepares the model,
        optimizers, and schedulers with the accelerator.

        Raises:
            ValueError: If ``do_train=True`` but
                :meth:`Model.configure_optimizers` returned no optimizers.
        """
        if not self.do_train:
            logger.info("do_train=False. Skipping optimizers and preparing model for inference.")
            prepared = self.accelerator.prepare(self.model)
            if not isinstance(prepared, tuple):
                prepared = (prepared,)
            self.model = prepared[0]
            return

        if self.sync_batch_norm and self.accelerator.num_processes > 1:
            logger.info("Converting model to SyncBatchNorm...")
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        configure_optim = self.model.configure_optimizers
        sig = inspect.signature(configure_optim)
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

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
        prepared = self.accelerator.prepare(*to_prepare)
        if not isinstance(prepared, tuple):
            prepared = (prepared,)

        self.model = prepared[0]

        # Splice the prepared optimizers and schedulers back out of the
        # flat tuple returned by accelerator.prepare.
        offset = 1
        if self.optimizers:
            self.optimizers = list(prepared[offset: offset + len(self.optimizers)])
            offset += len(self.optimizers)

        if self.schedulers:
            prepared_schedulers = prepared[offset:]
            for i, prep_sched in enumerate(prepared_schedulers):
                self.schedulers[i]['scheduler'] = prep_sched

    def _standardize_optimizers(self, opt_conf: Any) -> None:
        """Normalize the return of ``configure_optimizers`` into flat lists.

        Populates ``self.optimizers`` and ``self.schedulers`` from any of the
        supported return formats:

        - ``None`` (inference only).
        - A single :class:`~torch.optim.Optimizer`.
        - A dict with an ``"optimizer"`` key and optional
          ``"scheduler"`` / ``"lr_scheduler"`` key.
        - A list of such dicts.
        - A tuple ``(optimizers, schedulers)`` where each may be a single
          object or a list.
        - A plain list of optimizers (no schedulers).

        Each scheduler is stored as a config dict with keys ``scheduler``,
        ``strategy`` (``"epoch"`` by default), ``interval`` (``1`` by
        default), and ``monitor`` (``None`` by default).

        Args:
            opt_conf: The raw return value from
                :meth:`Model.configure_optimizers`.

        Raises:
            ValueError: If a dict is missing the ``"optimizer"`` key, if a
                scheduler config dict is missing the ``"scheduler"`` key, or
                if a ``ReduceLROnPlateau`` scheduler is given without a
                ``monitor`` key.
            TypeError: If ``opt_conf`` has an unsupported type, or if any
                element expected to be an optimizer is not one.
        """
        raw_optimizers: List[Any] = []
        raw_schedulers: List[Any] = []

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

            elif (len(opt_conf) == 2
                  and isinstance(opt_conf[0], torch.optim.Optimizer)
                  and not isinstance(opt_conf[1], torch.optim.Optimizer)):
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

            std_sched: Dict[str, Any] = {
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
        """The unwrapped model."""
        return self.accelerator.unwrap_model(self.model)

    @property
    def device(self) -> torch.device:
        """The device used by the accelerator."""
        return self.accelerator.device

    @property
    def is_main_process(self) -> bool:
        """Whether this process is the main process."""
        return self.accelerator.is_main_process

    @property
    def num_processes(self) -> int:
        """Total number of processes."""
        return self.accelerator.num_processes

    @property
    def sync_gradients(self) -> bool:
        """Whether gradients are being synced across processes right now."""
        return self.accelerator.sync_gradients

    @property
    def use_distributed(self) -> bool:
        """Whether distributed training is in use."""
        return self.accelerator.use_distributed

    @property
    def local_process_index(self) -> int:
        """Index of this process on the local machine."""
        return self.accelerator.local_process_index

    # ------------------------------------------------------------------
    # Context managers + distributed utils
    # ------------------------------------------------------------------

    def autocast(self):
        """Return the accelerator's autocast context manager."""
        return self.accelerator.autocast()

    def accumulate(self):
        """Return the accelerator's gradient-accumulation context manager."""
        return self.accelerator.accumulate(self.model)

    def wait(self) -> None:
        """Block until all processes reach this point."""
        self.accelerator.wait_for_everyone()

    def print(self, *args, **kwargs) -> None:
        """Print only on the main process."""
        self.accelerator.print(*args, **kwargs)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a tensor across all processes."""
        return self.accelerator.gather(tensor)

    def gather_for_metrics(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a tensor across processes for metric computation."""
        return self.accelerator.gather_for_metrics(tensor)

    def reduce(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Reduce a tensor across processes.

        Args:
            tensor: The tensor to reduce.
            reduction: One of ``"mean"``, ``"sum"``, or ``"none"``.

        Returns:
            The reduced tensor.
        """
        return self.accelerator.reduce(tensor, reduction=reduction)

    def free_memory(self) -> None:
        """Run garbage collection and empty the CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Training primitives
    # ------------------------------------------------------------------

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Run the backward pass through the accelerator.

        Args:
            loss: The loss tensor to backpropagate.
            **kwargs: Forwarded to ``accelerator.backward``.
        """
        self.accelerator.backward(loss, **kwargs)

    def clip_gradients(self) -> None:
        """Clip gradients if ``gradient_clip_value`` is set."""
        if self.gradient_clip_value is None:
            return
        if self.gradient_clip_algorithm == "value":
            self.accelerator.clip_grad_value_(self.model.parameters(), self.gradient_clip_value)
        else:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)

    def optimizer_zero_grad(self, idx: int, set_to_none: bool = True) -> None:
        """Zero gradients of the optimizer at ``idx``."""
        self.optimizers[idx].zero_grad(set_to_none=set_to_none)

    def optimizers_zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients of all optimizers."""
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def optimizer_step(self, idx: int) -> None:
        """Step the optimizer at ``idx``."""
        self.optimizers[idx].step()

    def optimizers_step(self) -> None:
        """Step all optimizers."""
        for opt in self.optimizers:
            opt.step()

    def scheduler_step(self, idx: int) -> None:
        """Step the scheduler at ``idx``.

        If the scheduler config has a ``monitor`` key, the corresponding
        value is read from :attr:`monitor` and passed to
        ``scheduler.step(value)`` (required for ``ReduceLROnPlateau``).

        Args:
            idx: Index into :attr:`schedulers`.

        Raises:
            KeyError: If the scheduler expects a ``monitor`` key that is not
                present in :attr:`monitor`.
        """
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

    def schedulers_step(self, strategy: str) -> None:
        """Step all schedulers matching the given strategy.

        Args:
            strategy: ``"step"`` or ``"epoch"``. Only schedulers whose
                ``strategy`` matches are stepped, and only when the current
                step/epoch counter is a multiple of their ``interval``.
        """
        counter = self.step if strategy == "step" else self.epoch
        for i, sched_dict in enumerate(self.schedulers):
            if sched_dict['strategy'] == strategy:
                if counter % sched_dict['interval'] == 0:
                    self.scheduler_step(i)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def should_validate(self) -> bool:
        """Return whether validation should run at the current step/epoch."""
        if not self.do_val or self.val_strategy == "no":
            return False
        counter = self.step if self.val_strategy == "step" else self.epoch
        return counter >= self.val_start and counter % self.val_interval == 0

    # ------------------------------------------------------------------
    # Loop: run (orchestrator)
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run training, validation, testing, and prediction as configured."""
        if self.do_train:
            self.run_train()
        if self.do_val:
            self.run_val()
        if self.do_test:
            self.run_test()
        if self.do_predict:
            self.run_predict()

    def run_train(self) -> None:
        """Run the full training loop.

        Iterates over epochs until ``max_epochs`` is reached or training is
        early-stopped via :attr:`should_stop` or ``max_steps``. Within each
        epoch, iterates over the training dataloader, accumulating gradients
        and stepping optimizers/schedulers at gradient-sync boundaries.

        Lifecycle hooks called (in order): ``on_train_start``,
        ``on_train_epoch_start``, ``on_train_substep_start``,
        ``on_train_substep_end``, ``on_train_step_start``,
        ``on_train_step_end``, ``on_train_epoch_end``, ``on_train_end``.
        Validation runs inline when :meth:`should_validate` is true.
        """
        self.on_train_start()

        epoch = 0
        while self.max_epochs is None or epoch < self.max_epochs:
            self.model.train()
            self.on_train_epoch_start()

            epoch_completed = True
            has_batch = False
            for batch_idx, batch in enumerate(self.train_dataloader):
                has_batch = True
                with self.accumulate():
                    self.on_train_substep_start(batch, batch_idx)
                    outputs = self.train_step(batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                    self.backward(loss)
                    self.on_train_substep_end(outputs, batch, batch_idx)

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

                    if self.should_stop or (self.max_steps is not None and self.step >= self.max_steps):
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

            if self.should_stop or (self.max_steps is not None and self.step >= self.max_steps):
                break

            epoch += 1

        self.on_train_end()

    def run_val(self) -> None:
        """Run the validation loop.

        Iterates over the validation dataloader (up to ``val_steps`` batches),
        calling :meth:`val_step` under ``torch.no_grad()``. Lifecycle hooks:
        ``on_val_start``, ``on_val_step_start``, ``on_val_step_end``,
        ``on_val_end``.
        """
        if self.val_dataloader is None:
            return
        self.model.eval()
        self.on_val_start()
        for batch_idx, batch in enumerate(self.val_dataloader):
            if self.val_steps is not None and batch_idx >= self.val_steps:
                break
            self.on_val_step_start(batch, batch_idx)
            with torch.no_grad():
                outputs = self.val_step(batch)
            self.on_val_step_end(outputs, batch, batch_idx)
        self.on_val_end()

    def run_test(self) -> None:
        """Run the testing loop.

        Iterates over the test dataloader (up to ``test_steps`` batches),
        calling :meth:`test_step` under ``torch.no_grad()``. Lifecycle hooks:
        ``on_test_start``, ``on_test_step_start``, ``on_test_step_end``,
        ``on_test_end``.
        """
        if self.test_dataloader is None:
            return
        self.model.eval()
        self.on_test_start()
        for batch_idx, batch in enumerate(self.test_dataloader):
            if self.test_steps is not None and batch_idx >= self.test_steps:
                break
            self.on_test_step_start(batch, batch_idx)
            with torch.no_grad():
                outputs = self.test_step(batch)
            self.on_test_step_end(outputs, batch, batch_idx)
        self.on_test_end()

    def run_predict(self) -> None:
        """Run the prediction loop.

        Iterates over the prediction dataloader (up to ``predict_steps``
        batches), calling :meth:`predict_step` under ``torch.no_grad()``.
        Lifecycle hooks: ``on_predict_start``, ``on_predict_step_start``,
        ``on_predict_step_end``, ``on_predict_end``.
        """
        if self.predict_dataloader is None:
            return
        self.model.eval()
        self.on_predict_start()
        for batch_idx, batch in enumerate(self.predict_dataloader):
            if self.predict_steps is not None and batch_idx >= self.predict_steps:
                break
            self.on_predict_step_start(batch, batch_idx)
            with torch.no_grad():
                outputs = self.predict_step(batch)
            self.on_predict_step_end(outputs, batch, batch_idx)
        self.on_predict_end()

    # ------------------------------------------------------------------
    # Step methods (user implements)
    # ------------------------------------------------------------------

    def train_step(self, batch) -> torch.Tensor:
        """Compute and return the training loss for a batch.

        May return either a loss tensor directly, or a dict containing at
        least a ``"loss"`` key. When a dict is returned, the engine extracts
        ``outputs["loss"]`` for the backward pass and forwards the full
        ``outputs`` dict to :meth:`on_train_substep_end`.

        Args:
            batch: A batch from the training dataloader.

        Returns:
            The loss tensor, or a dict with a ``"loss"`` key.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError("train_step must be implemented to train.")

    def val_step(self, batch) -> Optional[Dict[str, Any]]:
        """Compute and return validation outputs for a batch.

        Args:
            batch: A batch from the validation dataloader.

        Returns:
            A dict of outputs (e.g. ``{"val_loss": ...}``), or ``None``.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError("val_step must be implemented to validate.")

    def test_step(self, batch) -> Optional[Dict[str, Any]]:
        """Compute and return test outputs for a batch.

        Args:
            batch: A batch from the test dataloader.

        Returns:
            A dict of outputs (e.g. ``{"test_loss": ...}``), or ``None``.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError("test_step must be implemented to test.")

    def predict_step(self, batch) -> Any:
        """Compute and return predictions for a batch.

        Args:
            batch: A batch from the prediction dataloader.

        Returns:
            The model's predictions for the batch.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError("predict_step must be implemented to predict.")

    # ------------------------------------------------------------------
    # Lifecycle hooks (all no-ops by default)
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Called once at the beginning of training, before the first epoch."""
        pass

    def on_train_epoch_start(self) -> None:
        """Called at the beginning of each training epoch."""
        pass

    def on_train_substep_start(self, batch, batch_idx) -> None:
        """Called at the start of each micro-batch (gradient-accumulation substep).

        Args:
            batch: The current micro-batch.
            batch_idx: Index of the current micro-batch within the epoch.
        """
        pass

    def on_train_substep_end(self, outputs, batch, batch_idx) -> None:
        """Called at the end of each micro-batch (gradient-accumulation substep).

        Args:
            outputs: The full return value of :meth:`train_step` (a loss
                tensor or a dict containing a ``"loss"`` key).
            batch: The current micro-batch.
            batch_idx: Index of the current micro-batch within the epoch.
        """
        pass

    def on_train_step_start(self) -> None:
        """Called at the start of each optimizer step (after gradient sync)."""
        pass

    def on_train_step_end(self) -> None:
        """Called at the end of each optimizer step (after gradient sync)."""
        pass

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        pass

    def on_train_end(self) -> None:
        """Called once at the end of training, after the last epoch."""
        pass

    def on_val_start(self) -> None:
        """Called at the beginning of the validation loop."""
        pass

    def on_val_step_start(self, batch, batch_idx) -> None:
        """Called before each validation batch.

        Args:
            batch: The current validation batch.
            batch_idx: Index of the current batch within the validation loop.
        """
        pass

    def on_val_step_end(self, outputs, batch, batch_idx) -> None:
        """Called after each validation batch.

        Args:
            outputs: The return value of :meth:`val_step`.
            batch: The current validation batch.
            batch_idx: Index of the current batch within the validation loop.
        """
        pass

    def on_val_end(self) -> None:
        """Called at the end of the validation loop."""
        pass

    def on_test_start(self) -> None:
        """Called at the beginning of the testing loop."""
        pass

    def on_test_step_start(self, batch, batch_idx) -> None:
        """Called before each test batch.

        Args:
            batch: The current test batch.
            batch_idx: Index of the current batch within the testing loop.
        """
        pass

    def on_test_step_end(self, outputs, batch, batch_idx) -> None:
        """Called after each test batch.

        Args:
            outputs: The return value of :meth:`test_step`.
            batch: The current test batch.
            batch_idx: Index of the current batch within the testing loop.
        """
        pass

    def on_test_end(self) -> None:
        """Called at the end of the testing loop."""
        pass

    def on_predict_start(self) -> None:
        """Called at the beginning of the prediction loop."""
        pass

    def on_predict_step_start(self, batch, batch_idx) -> None:
        """Called before each prediction batch.

        Args:
            batch: The current prediction batch.
            batch_idx: Index of the current batch within the prediction loop.
        """
        pass

    def on_predict_step_end(self, outputs, batch, batch_idx) -> None:
        """Called after each prediction batch.

        Args:
            outputs: The return value of :meth:`predict_step`.
            batch: The current prediction batch.
            batch_idx: Index of the current batch within the prediction loop.
        """
        pass

    def on_predict_end(self) -> None:
        """Called at the end of the prediction loop."""
        pass
