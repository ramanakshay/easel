import math
import shutil
import logging
import re
import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

# Internal imports
from .module import Module
from .data_module import DataModule
from .runner import Runner

logger = logging.getLogger(__name__)

class Trainer(Runner):
    """
    A professional, generic training engine integrating Hugging Face Accelerate.
    Supports flexible logging, checkpointing, and resuming strategies for SL and RL tasks.
    """
    def __init__(self,
                 model: Module,
                 data: DataModule,
                 # --- Task Flags ---
                 do_collect: bool = True,
                 do_train: bool = True,
                 do_val: bool = True,
                 do_test: bool = True,

                 # --- Trainer Control ---
                 max_epochs: Optional[int] = None,
                 max_steps: Optional[int] = None,

                 collect_steps_per_epoch: Optional[int] = None,
                 train_steps_per_epoch: Optional[int] = None,
                 val_steps_per_epoch: Optional[int] = None,
                 test_steps_per_epoch: Optional[int] = None,

                 # --- Execution Strategies ---
                 train_strategy: str = "epoch",
                 train_start: int = 0,
                 train_interval: int = 1,

                 val_strategy: str = "epoch",
                 val_start: int = 0,
                 val_interval: int = 1,

                 # --- Logging Configuration ---
                 log_with: Union[str, List[str], None] = None,
                 log_strategy: str = "epoch",
                 log_start: int = 0,
                 log_interval: int = 1,
                 log_config: Optional[Dict[str, Any]] = None,

                 # --- Saving & Resuming Configuration ---
                 output_dir: str = "outputs",
                 resume_from_checkpoint: Optional[str] = None,
                 save_strategy: str = "epoch",
                 save_start: int = 0,
                 save_interval: int = 1,
                 save_config: Optional[Dict[str, Any]] = None,

                 # --- Runner/Accelerator Configs ---
                 experiment_name: str = "default_experiment",
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 dataloader_config: Optional[Dict[str, Any]] = None,
                 mixed_precision: str = "no",
                 grad_accum_steps: int = 1,
                 accelerator_config: Optional[Dict[str, Any]] = None,
                 seed: int = 42,
                 deterministic: bool = False,
                 use_tf32: bool = False,
                 cudnn_benchmark: bool = False,
                 compile: bool = False,
                 compile_config: Optional[Dict[str, Any]] = None,
                 grad_clip_value: float = 0.0,
                 grad_clip_algorithm: str = "norm",
                 sync_batch_norm: bool = False):

        self.do_collect = do_collect

        # 2. Flow Control & Limits
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.max_steps = max_steps

        self.collect_steps_per_epoch = collect_steps_per_epoch
        self.train_steps_per_epoch = train_steps_per_epoch
        self.val_steps_per_epoch = val_steps_per_epoch
        self.test_steps_per_epoch = test_steps_per_epoch

        self.train_strategy = train_strategy
        self.train_start = train_start
        self.train_interval = train_interval

        self.val_strategy = val_strategy
        self.val_start = val_start
        self.val_interval = val_interval

        self.log_strategy = log_strategy
        self.log_start = log_start
        self.log_interval = log_interval

        self.save_strategy = save_strategy
        self.save_start = save_start
        self.save_interval = save_interval

        # 3. Saving & Resuming Logic
        self.resume_from_checkpoint = resume_from_checkpoint

        self.save_config = save_config or {}
        self.save_config.setdefault("limit", None)
        self.save_config.setdefault("monitor", None)
        self.save_config.setdefault("mode", "min")
        self.save_config.setdefault("only_model", False)
        self.save_config.setdefault("safetensors", False)

        if self.save_config["mode"] == "min":
            self.best_metric = float("inf")
        else:
            self.best_metric = float("-inf")

        # 4. Internal State
        # Monitor: Controlled MANUALLY by user (for schedulers/checkpoints/best model)
        self.monitor = {}
        # Log Buffer: Controlled by self.log() (for Visualization)
        self._log_buffer = {}

        self.step = 0
        self.epoch = 0

        super().__init__(
            model=model,
            data=data,
            do_train=do_train,
            do_val=do_val,
            do_test=do_test,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            dataloader_config=dataloader_config,
            mixed_precision=mixed_precision,
            grad_accum_steps=grad_accum_steps,
            log_with=log_with,
            project_dir=output_dir,
            project_name=experiment_name,
            log_config=log_config,
            accelerator_config=accelerator_config,
            seed=seed,
            deterministic=deterministic,
            use_tf32=use_tf32,
            cudnn_benchmark=cudnn_benchmark,
            compile=compile,
            compile_config=compile_config,
            grad_clip_value=grad_clip_value,
            grad_clip_algorithm=grad_clip_algorithm,
            sync_batch_norm=sync_batch_norm
        )

        self._auto_detect_tasks()

    def _auto_detect_tasks(self):
        """Automatically disables tasks if their abstract methods are not implemented."""
        def is_overridden(method_name):
            method = getattr(self, method_name)
            base_method = getattr(Trainer, method_name)
            return method.__func__ is not base_method

        if self.do_collect and not is_overridden("collect_step"):
            logger.warning("do_collect=True but 'collect_step' not implemented. Disabling collection.")
            self.do_collect = False

        if self.do_train and not is_overridden("train_step"):
            logger.warning("do_train=True but 'train_step' not implemented. Disabling training.")
            self.do_train = False

        if self.do_val and not is_overridden("val_step"):
            logger.warning("do_val=True but 'val_step' not implemented. Disabling validation.")
            self.do_val = False

        if self.do_test and not is_overridden("test_step"):
            logger.warning("do_test=True but 'test_step' not implemented. Disabling testing.")
            self.do_test = False

    def setup(self):
        super().setup()

        # 1. Establish Step Counts
        if self.do_collect:
            if not self.collect_steps_per_epoch or self.collect_steps_per_epoch <= 0:
                self.do_collect = False

        if self.do_train:
            if self.train_steps_per_epoch is None:
                try:
                    length = len(self.train_dataloader)
                    self.train_steps_per_epoch = math.ceil(length / self.grad_accum_steps)
                except (TypeError, AttributeError):
                    raise ValueError("train_dataloader has no len(). Set 'train_steps_per_epoch' manually.")
            if self.train_steps_per_epoch <= 0:
                self.do_train = False

        if self.do_val:
            if self.val_steps_per_epoch is None:
                try:
                    self.val_steps_per_epoch = len(self.val_dataloader)
                except (TypeError, AttributeError):
                    raise ValueError("val_dataloader has no len(). Set 'val_steps_per_epoch' manually.")

        if self.do_test:
            if self.test_steps_per_epoch is None:
                try:
                    self.test_steps_per_epoch = len(self.test_dataloader)
                except (TypeError, AttributeError):
                    raise ValueError("test_dataloader has no len(). Set 'test_steps_per_epoch' manually.")

        # 2. Establish Total Duration
        steps_per_epoch = self.collect_steps_per_epoch if self.do_collect else self.train_steps_per_epoch
        if steps_per_epoch == 0:
             steps_per_epoch = 1

        if self.max_epochs is None and self.max_steps is None:
            self.max_epochs = 1
            self.max_steps = steps_per_epoch
        elif self.max_steps is None:
            self.max_steps = self.max_epochs * steps_per_epoch
        elif self.max_epochs is None:
            self.max_epochs = math.ceil(self.max_steps / steps_per_epoch)

        # 3. Handle Resuming
        if self.resume_from_checkpoint:
            self.load_checkpoint(self.resume_from_checkpoint)

    # =========================================================================
    # TRAINER RUN LOOP
    # =========================================================================
    def run(self):
        if self.do_collect:
            self.run_collect()
        elif self.do_train:
            self.run_train()
        elif self.do_val:
            self.run_val()

        if self.do_test:
            self.run_test()

        self.finish()

    def run_collect(self):
        for epoch in range(self.epoch, self.max_epochs):
            self.run_collect_epoch(is_main_task=True)
            self.run_interleaved_tasks(strategy="epoch")
            if self.max_steps is not None and self.step >= self.max_steps:
                break

    def run_train(self):
        for epoch in range(self.epoch, self.max_epochs):
            self.run_train_epoch(is_main_task=True)
            self.run_interleaved_tasks(strategy="epoch")
            if self.max_steps is not None and self.step >= self.max_steps:
                break

    def run_val(self):
        self.run_val_epoch()

    def run_test(self):
        self.run_test_epoch()

    # =========================================================================
    # EPOCH LOOPS
    # =========================================================================

    def run_collect_epoch(self, is_main_task=True):
        self.model.eval()

        start_step = 0
        if self.resume_from_checkpoint and is_main_task:
            if self.collect_steps_per_epoch > 0:
                start_step = self.step % self.collect_steps_per_epoch
                if start_step > 0:
                    logger.info(f"Resuming collection from epoch step {start_step}.")
            self.resume_from_checkpoint = None

        with torch.inference_mode():
            self.on_collect_epoch_start()

        for _ in range(start_step, self.collect_steps_per_epoch):
            with torch.inference_mode():
                self.on_collect_step_start()
                self.collect_step()
                self.on_collect_step_end()

            if is_main_task:
                self.step += 1
                self.run_interleaved_tasks(strategy="step")

                if self.max_steps is not None and self.step >= self.max_steps:
                    break

        with torch.inference_mode():
            self.on_collect_epoch_end()

        if is_main_task:
            self.epoch += 1

    def run_train_epoch(self, is_main_task=True):
        self.model.train()

        # 1. DDP Sampler handling
        if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        active_dataloader = self.train_dataloader
        batches_to_skip = 0

        # 2. Resume Logic
        if self.resume_from_checkpoint and is_main_task:
            total_batches = len(self.train_dataloader)
            total_substeps_done = self.step * self.grad_accum_steps
            batches_to_skip = total_substeps_done % total_batches

            if batches_to_skip > 0:
                logger.info(f"Resuming: Skipping first {batches_to_skip} batches in epoch {self.epoch}.")
                active_dataloader = self.accelerator.skip_first_batches(active_dataloader, batches_to_skip)

            self.resume_from_checkpoint = None

        self.on_train_epoch_start()

        # 3. Initialize Metrics
        step_loss = torch.zeros(1, device=self.accelerator.device)

        # Throughput: Use loader batch size (efficient)
        loader_bs = getattr(self.train_dataloader, "batch_size", None)
        per_device_batch_size = loader_bs if loader_bs is not None else self.train_batch_size

        accumulated_samples = 0
        num_micro_batches = 0
        train_batches_per_epoch = self.train_steps_per_epoch * self.grad_accum_steps

        # 4. Initialize Counters
        # [FIX] If we skipped batches, our local "epoch_step" should reflect that offset.
        epoch_step = batches_to_skip // self.grad_accum_steps

        step_start_time = time.perf_counter()

        for batch_idx, batch in enumerate(active_dataloader):
            # batch_idx here starts at 0 for the *active* loader.
            # If we skipped batches, the real data index is (batch_idx + batches_to_skip).
            real_batch_idx = batch_idx + batches_to_skip

            self.on_train_substep_start(batch, real_batch_idx)

            # 5. Forward & Accumulate
            with self.accumulate():
                outputs = self.train_step(batch, real_batch_idx)
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

                self.backward(loss / self.grad_accum_steps)

                step_loss += loss.detach()
                accumulated_samples += per_device_batch_size
                num_micro_batches += 1

            self.on_train_substep_end(outputs, batch, real_batch_idx)

            # 6. Optimizer Step (Sync)
            if self.sync_gradients:
                # --- Metrics Calculation ---
                now = time.perf_counter()
                delta = now - step_start_time
                total_samples = accumulated_samples * self.num_processes
                throughput = total_samples / delta if delta > 0 else 0.0
                step_start_time = time.perf_counter() # Reset immediately

                # --- Hooks ---
                self.on_train_step_start(epoch_step)

                grad_norm = self.gradient_clip_step()
                self.optimizers_step(zero_grad=True)

                raw_loss = step_loss / num_micro_batches

                metrics = {
                    "loss": raw_loss.detach(),
                    "grad_norm": grad_norm,
                    "lr": self.get_lr(),
                    "throughput": throughput
                }

                # Log to buffer
                self.log(metrics)

                # Pass metrics to user hook
                self.on_train_step_end(metrics, epoch_step)

                # --- Cleanup ---
                step_loss.zero_()
                accumulated_samples = 0
                num_micro_batches = 0

                epoch_step += 1
                if is_main_task:
                    self.step += 1

                self.schedulers_step(self.step, strategy="step")

                # --- Interleaved Tasks ---
                if is_main_task:
                    self.run_interleaved_tasks(strategy="step")
                    if self.max_steps is not None and self.step >= self.max_steps:
                        break

            # 7. Safety Break
            if (real_batch_idx + 1) >= train_batches_per_epoch:
                break

        self.on_train_epoch_end()
        if is_main_task:
            self.epoch += 1
        self.schedulers_step(self.epoch, strategy="epoch")


    def run_val_epoch(self):
        self.model.eval()
        with torch.inference_mode():
            self.on_val_epoch_start()
            for batch_idx, batch in enumerate(self.val_dataloader):
                self.on_val_step_start(batch, batch_idx)
                outputs = self.val_step(batch, batch_idx)
                self.on_val_step_end(outputs, batch, batch_idx)
                if batch_idx + 1 >= self.val_steps_per_epoch:
                    break
            self.on_val_epoch_end()

    def run_test_epoch(self):
        self.model.eval()
        with torch.inference_mode():
            self.on_test_epoch_start()
            for batch_idx, batch in enumerate(self.test_dataloader):
                self.on_test_step_start(batch, batch_idx)
                outputs = self.test_step(batch, batch_idx)
                self.on_test_step_end(outputs, batch, batch_idx)
                if batch_idx + 1 >= self.test_steps_per_epoch:
                    break
            self.on_test_epoch_end()

    # =========================================================================
    # INTERLEAVED TASKS
    # =========================================================================
    def run_interleaved_tasks(self, strategy="step"):
        counter = self.step if strategy == "step" else self.epoch

        if self.do_collect and self.do_train and self.train_strategy == strategy:
             if counter >= self.train_start and counter % self.train_interval == 0:
                self.run_train_epoch(is_main_task=False)
                # Ensure we return to eval mode for collection
                self.model.eval()

        if self.do_val and self.val_strategy == strategy:
            if counter >= self.val_start and counter % self.val_interval == 0:
                self.run_val_epoch()
                # Return to train mode only if we are in SL mode
                if self.do_train and not self.do_collect:
                    self.model.train()

        if self.should_log:
            self.run_log_step()

        if self.save_strategy == strategy:
            if counter >= self.save_start and counter % self.save_interval == 0:
                self.run_save_step()

    # =========================================================================
    # LOGGING & MONITORING
    # =========================================================================

    @property
    def should_save(self) -> bool:
        """Returns True if the current step/epoch matches the save schedule."""
        counter = self.step if self.save_strategy == "step" else self.epoch
        return (counter >= self.save_start) and (counter % self.save_interval == 0)

    @property
    def should_log(self) -> bool:
        if not self.log_with:
            return False
        counter = self.step if self.log_strategy == "step" else self.epoch
        return (counter >= self.log_start) and (counter % self.log_interval == 0)

    def log(self, metrics: Dict[str, Any], commit: bool = False):
        """
        Updates the internal log buffer.
        Note: self.monitor is NOT updated here. User must update it manually.
        """
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                clean_metrics[k] = v.item()
            elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (int, float)):
                 clean_metrics[k] = v[0]
            else:
                clean_metrics[k] = v

        # Only update the buffer for visualization
        self._log_buffer.update(clean_metrics)

        if commit:
            self.run_log_step()

    def run_log_step(self):
        if not self._log_buffer:
            return
        super().log(self._log_buffer, step=self.step)
        self._log_buffer.clear()

    # =========================================================================
    # SAVING & LOADING
    # =========================================================================

    def run_save_step(self):
        is_best = False
        monitor_metric = self.save_config["monitor"]

        if monitor_metric:
            if monitor_metric in self.monitor:
                current_val = self.monitor[monitor_metric]
                mode = self.save_config["mode"]

                improved = (mode == "min" and current_val < self.best_metric) or \
                           (mode == "max" and current_val > self.best_metric)

                if improved:
                    self.best_metric = current_val
                    is_best = True
                    logger.info(f"New best model found! ({monitor_metric}: {current_val})")
            else:
                logger.warning(f"Save monitor '{monitor_metric}' not found in metrics. Saving latest checkpoint anyway.")

        folder_name = f"checkpoint_epoch{self.epoch}_step{self.step}"
        latest_path = os.path.join(self.output_dir, folder_name)

        self.save_checkpoint(latest_path)

        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoint_best")
            if os.path.exists(best_path):
                shutil.rmtree(best_path)
            shutil.copytree(latest_path, best_path)
            logger.info(f"Copied best model to {best_path}")

    def save_checkpoint(self, path: Optional[str] = None, **kwargs):
        if path is None:
            path = os.path.join(self.output_dir, f"checkpoint_epoch{self.epoch}_step{self.step}")

        safe_serialization = self.save_config.get("safetensors", False)
        if self.save_config["only_model"]:
            self.save_model(path, safe_serialization=safe_serialization, **kwargs)
        else:
            self.save_state(path, safe_serialization=safe_serialization, **kwargs)

        # trainer_state needs safe types from self.monitor
        # We assume the user has been responsible putting valid JSON types into self.monitor
        trainer_state = {
            "epoch": self.epoch,
            "step": self.step,
            "monitor": self.monitor,
            "best_metric": self.best_metric
        }

        # Hook: Allow user to inject custom state
        self.on_save_checkpoint(trainer_state)

        os.makedirs(path, exist_ok=True)
        state_path = os.path.join(path, "trainer_state.json")
        try:
            with open(state_path, "w") as f:
                json.dump(trainer_state, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save trainer_state.json: {e}")

        logger.info(f"Saved checkpoint to {path}")
        self._rotate_checkpoints()

    def load_checkpoint(self, path: str):
        logger.info(f"Loading checkpoint from: {path}")

        # Intelligent Resume: Check if we have a full state or just weights
        # Accelerate usually creates 'random_states*' or 'optimizer*' files for full state
        has_state = False
        if os.path.exists(path):
            files = os.listdir(path)
            # Simple heuristic: look for optimizer/rng files usually present in save_state
            if any(f.startswith("optimizer") or f.startswith("random_states") or f.startswith("scheduler") for f in files):
                has_state = True

        if has_state:
            self.load_state(path)
        else:
            logger.warning(f"Checkpoint '{path}' seems to lack optimizer/RNG state (only_model=True?). Loading weights only.")
            self.load_model(path)

        state_path = os.path.join(path, "trainer_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
                self.epoch = state.get("epoch", self.epoch)
                self.step = state.get("step", self.step)
                self.monitor = state.get("monitor", self.monitor)
                self.best_metric = state.get("best_metric", self.best_metric)

                # Hook: Allow user to restore custom state
                self.on_load_checkpoint(state)

                logger.info(f"Restored Trainer State: Epoch {self.epoch}, Step {self.step}, Best {self.best_metric}")
            except Exception as e:
                logger.warning(f"Failed to load trainer_state.json: {e}")
        else:
            # Fallback: Parse directory name
            folder_name = os.path.basename(path.rstrip('/'))
            match = re.search(r"epoch(\d+)_step(\d+)", folder_name)
            if match:
                self.epoch = int(match.group(1))
                self.step = int(match.group(2))
                logger.info(f"Restored counters from filename: Epoch {self.epoch}, Step {self.step}")
            else:
                logger.warning(f"No state found. Keeping current counters (Epoch {self.epoch}, Step {self.step}).")

    def _rotate_checkpoints(self):
        limit = self.save_config["limit"]
        if limit is None:
            return

        # Sort by (epoch, step) extracted from filename, NOT timestamp
        glob_pattern = "checkpoint_epoch*_step*"
        try:
            checkpoints = list(Path(self.output_dir).glob(glob_pattern))

            def sort_key(p):
                match = re.search(r"epoch(\d+)_step(\d+)", p.name)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
                return (0, 0)

            checkpoints.sort(key=sort_key)
        except Exception:
            return

        if len(checkpoints) > limit:
            to_delete = checkpoints[: len(checkpoints) - limit]
            for p in to_delete:
                shutil.rmtree(p, ignore_errors=True)
                logger.info(f"Deleted old checkpoint: {p}")

    def _get_batch_size(self, batch) -> int:
        """Robustly finds the first tensor in the batch to determine size."""
        if isinstance(batch, torch.Tensor):
            return batch.size(0)

        if isinstance(batch, (list, tuple)):
            for item in batch:
                res = self._get_batch_size(item)
                if res > 0: return res

        if isinstance(batch, dict):
            for v in batch.values():
                res = self._get_batch_size(v)
                if res > 0: return res

        return 1

    # =========================================================================
    # USER EXTENSION POINTS
    # =========================================================================
    def train_step(self, batch, batch_idx): raise NotImplementedError
    def val_step(self, batch, batch_idx): raise NotImplementedError
    def test_step(self, batch, batch_idx): raise NotImplementedError
    def collect_step(self): raise NotImplementedError

    def on_collect_epoch_start(self): pass
    def on_collect_epoch_end(self): pass
    def on_train_epoch_start(self): pass
    def on_train_epoch_end(self): pass
    def on_val_epoch_start(self): pass
    def on_val_epoch_end(self): pass
    def on_test_epoch_start(self): pass
    def on_test_epoch_end(self): pass

    def on_collect_step_start(self): pass
    def on_collect_step_end(self): pass
    def on_train_step_start(self, batch_idx): pass
    def on_train_step_end(self, metrics, batch_idx): pass
    def on_train_substep_start(self, batch, batch_idx): pass
    def on_train_substep_end(self, outputs, batch, batch_idx): pass
    def on_val_step_start(self, batch, batch_idx): pass
    def on_val_step_end(self, outputs, batch, batch_idx): pass
    def on_test_step_start(self, batch, batch_idx): pass
    def on_test_step_end(self, outputs, batch, batch_idx): pass

    # NEW: Persistence Hooks
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Called before saving trainer_state.json. Inject custom state here."""
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Called after loading trainer_state.json. Restore custom state here."""
        pass
