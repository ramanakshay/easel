import logging
import time
import torch
from typing import Any, Dict, List, Optional
from tqdm.auto import tqdm

# Internal imports
from .module import Module
from .data_module import DataModule
from .runner import Runner

logger = logging.getLogger(__name__)

class Evaluator(Runner):
    """
    A streamlined engine for inference, validation, testing, and data collection.

    API Consistency:
    - Uses the same hook names as Trainer (on_val_epoch_start, etc.)
    - Uses the same step limits (val_steps_per_epoch)
    - Returns aggregated metrics from .run()
    """
    def __init__(self,
                 model: Module,
                 data: DataModule,
                 # --- Task Flags ---
                 do_collect: bool = False,
                 do_val: bool = True,
                 do_test: bool = False,

                 # --- Step Limits ---
                 # Renamed back to *_steps_per_epoch for consistency with Trainer
                 collect_steps_per_epoch: Optional[int] = None,
                 val_steps_per_epoch: Optional[int] = None,
                 test_steps_per_epoch: Optional[int] = None,

                 # --- Runner Configs ---
                 experiment_name: str = "eval_run",
                 eval_batch_size: int = 32,
                 dataloader_config: Optional[Dict[str, Any]] = None,
                 mixed_precision: str = "no",
                 accelerator_config: Optional[Dict[str, Any]] = None,
                 seed: int = 42,
                 deterministic: bool = False,
                 use_tf32: bool = False,
                 cudnn_benchmark: bool = False,
                 compile: bool = False,
                 compile_config: Optional[Dict[str, Any]] = None):

        self.do_collect = do_collect
        self.collect_steps_per_epoch = collect_steps_per_epoch
        self.val_steps_per_epoch = val_steps_per_epoch
        self.test_steps_per_epoch = test_steps_per_epoch

        super().__init__(
            model=model,
            data=data,
            do_train=False,
            do_val=do_val,
            do_test=do_test,
            train_batch_size=1,
            eval_batch_size=eval_batch_size,
            dataloader_config=dataloader_config,
            mixed_precision=mixed_precision,
            log_with=None,
            project_dir="outputs_eval",
            project_name=experiment_name,
            log_config=None,
            accelerator_config=accelerator_config,
            seed=seed,
            deterministic=deterministic,
            use_tf32=use_tf32,
            cudnn_benchmark=cudnn_benchmark,
            compile=compile,
            compile_config=compile_config,
        )

        self._auto_detect_tasks()

    def _auto_detect_tasks(self):
        def is_overridden(method_name):
            method = getattr(self, method_name)
            base_method = getattr(Evaluator, method_name)
            return method.__func__ is not base_method

        if self.do_collect and not is_overridden("collect_step"):
            logger.warning("do_collect=True but 'collect_step' not implemented. Disabling collection.")
            self.do_collect = False

        if self.do_val and not is_overridden("val_step"):
            logger.warning("do_val=True but 'val_step' not implemented. Disabling validation.")
            self.do_val = False

        if self.do_test and not is_overridden("test_step"):
            logger.warning("do_test=True but 'test_step' not implemented. Disabling testing.")
            self.do_test = False

    def setup(self):
        super().setup()

        # Resolve Step Counts
        if self.do_collect:
            if not self.collect_steps_per_epoch or self.collect_steps_per_epoch <= 0:
                logger.warning("do_collect=True but collect_steps_per_epoch is 0 or None. Disabling.")
                self.do_collect = False

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

    # =========================================================================
    # CORE EXECUTION
    # =========================================================================

    def run(self) -> Dict[str, Any]:
        results = {}

        if self.do_collect:
            logger.info("Starting Collection...")
            res = self.run_collect_epoch()
            if res: results.update(res)

        if self.do_val:
            logger.info("Starting Validation...")
            res = self.run_val_epoch()
            if res: results.update(res)

        if self.do_test:
            logger.info("Starting Testing...")
            res = self.run_test_epoch()
            if res: results.update(res)

        self.finish()
        return results

    # =========================================================================
    # TASK LOOPS (Renamed to match Trainer)
    # =========================================================================

    def run_collect_epoch(self) -> Optional[Dict[str, Any]]:
        self.model.eval()
        pbar = tqdm(total=self.collect_steps_per_epoch, desc="Collecting", disable=not self.accelerator.is_local_main_process)

        start_time = time.perf_counter()
        total_samples = 0

        with torch.inference_mode():
            self.on_collect_epoch_start()

            for step_idx in range(self.collect_steps_per_epoch):
                self.on_collect_step_start()
                outputs = self.collect_step()
                self.on_collect_step_end(outputs)

                total_samples += 1
                pbar.update(1)

            duration = time.perf_counter() - start_time
            throughput = total_samples / duration if duration > 0 else 0.0
            pbar.set_postfix({"steps/s": f"{throughput:.2f}"})
            pbar.close()

            return self.on_collect_epoch_end()

    def run_val_epoch(self) -> Optional[Dict[str, Any]]:
        self.model.eval()
        pbar = tqdm(total=self.val_steps_per_epoch, desc="Validation", disable=not self.accelerator.is_local_main_process)

        start_time = time.perf_counter()
        total_samples = 0

        with torch.inference_mode():
            self.on_val_epoch_start()

            for batch_idx, batch in enumerate(self.val_dataloader):
                self.on_val_step_start(batch, batch_idx)
                outputs = self.val_step(batch, batch_idx)
                self.on_val_step_end(outputs, batch, batch_idx)

                # Estimate batch size for throughput
                batch_size = 1
                try:
                    if isinstance(batch, torch.Tensor): batch_size = batch.size(0)
                    elif isinstance(batch, (list, tuple)): batch_size = batch[0].size(0)
                except: pass

                total_samples += batch_size
                pbar.update(1)

                if (batch_idx + 1) >= self.val_steps_per_epoch:
                    break

            duration = time.perf_counter() - start_time
            throughput = total_samples / duration if duration > 0 else 0.0
            pbar.set_postfix({"samples/s": f"{throughput:.2f}"})
            pbar.close()

            return self.on_val_epoch_end()

    def run_test_epoch(self) -> Optional[Dict[str, Any]]:
        self.model.eval()
        pbar = tqdm(total=self.test_steps_per_epoch, desc="Testing", disable=not self.accelerator.is_local_main_process)

        start_time = time.perf_counter()
        total_samples = 0

        with torch.inference_mode():
            self.on_test_epoch_start()

            for batch_idx, batch in enumerate(self.test_dataloader):
                self.on_test_step_start(batch, batch_idx)
                outputs = self.test_step(batch, batch_idx)
                self.on_test_step_end(outputs, batch, batch_idx)

                batch_size = 1
                try:
                    if isinstance(batch, torch.Tensor): batch_size = batch.size(0)
                    elif isinstance(batch, (list, tuple)): batch_size = batch[0].size(0)
                except: pass

                total_samples += batch_size
                pbar.update(1)

                if (batch_idx + 1) >= self.test_steps_per_epoch:
                    break

            duration = time.perf_counter() - start_time
            throughput = total_samples / duration if duration > 0 else 0.0
            pbar.set_postfix({"samples/s": f"{throughput:.2f}"})
            pbar.close()

            return self.on_test_epoch_end()

    # =========================================================================
    # USER EXTENSION POINTS (Aligned with Trainer)
    # =========================================================================

    # Steps
    def collect_step(self): raise NotImplementedError
    def val_step(self, batch, batch_idx): raise NotImplementedError
    def test_step(self, batch, batch_idx): raise NotImplementedError

    # Hooks
    def on_collect_epoch_start(self): pass
    def on_collect_epoch_end(self) -> Optional[Dict[str, Any]]: pass
    def on_collect_step_start(self): pass
    def on_collect_step_end(self, outputs): pass

    def on_val_epoch_start(self): pass
    def on_val_epoch_end(self) -> Optional[Dict[str, Any]]: pass
    def on_val_step_start(self, batch, batch_idx): pass
    def on_val_step_end(self, outputs, batch, batch_idx): pass

    def on_test_epoch_start(self): pass
    def on_test_epoch_end(self) -> Optional[Dict[str, Any]]: pass
    def on_test_step_start(self, batch, batch_idx): pass
    def on_test_step_end(self, outputs, batch, batch_idx): pass
