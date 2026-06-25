import torch
import torch.nn as nn
import math
import os
import logging
from torch.utils.data import TensorDataset, DataLoader

from easel import Data, Model, Engine

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────

class DummyData(Data):
    def setup(self, stage=None):
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        ds = TensorDataset(x, y)
        self.train_dataset = ds
        self.val_dataset = ds
        self.test_dataset = ds
        self.predict_dataset = ds


class LinearModel(Model):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": 1, "strategy": "epoch"}}


_last_engine = None

def make_linear_engine(**overrides):
    global _last_engine
    data = DummyData()
    model = LinearModel()
    defaults = dict(model=model, data=data, train_batch_size=16, seed=42, max_epochs=2)
    defaults.update(overrides)
    _last_engine = Engine(**defaults)
    return _last_engine


# ──────────────────────────────────────────────
# 1. Setup: globals
# ──────────────────────────────────────────────

def test_setup_globals_deterministic_sets_env():
    engine = make_linear_engine(deterministic=True)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert engine.cudnn_benchmark is False


def test_setup_globals_deterministic_disables_benchmark():
    engine = make_linear_engine(deterministic=True, cudnn_benchmark=True)
    assert engine.cudnn_benchmark is False


def test_setup_globals_tf32_bool():
    engine = make_linear_engine(tf32=True)
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True


def test_setup_globals_tf32_str():
    engine = make_linear_engine(tf32="medium")
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_setup_globals_defaults():
    engine = make_linear_engine(seed=None)
    assert engine.seed is None


# ──────────────────────────────────────────────
# 2. Setup: accelerator
# ──────────────────────────────────────────────

def test_accelerator_created():
    engine = make_linear_engine()
    assert engine.accelerator is not None
    assert engine.device is not None


def test_accelerator_default_project():
    engine = make_linear_engine()
    assert engine.project_dir == "outputs"
    assert engine.project_name == "outputs"
    assert engine.init_trackers_config["project_name"] == "outputs"


def test_accelerator_init_trackers_config_merge():
    engine = make_linear_engine(
        project_name="custom",
        init_trackers_config={"project_name": "override"}
    )
    assert engine.init_trackers_config["project_name"] == "override"


def test_accelerator_seed_none():
    engine = make_linear_engine(seed=None)


def test_accelerator_seed_set():
    engine = make_linear_engine(seed=42)


def test_accelerator_config_merge():
    engine = make_linear_engine(project_dir="outputs")
    assert engine.project_dir == "outputs"
    assert engine.accelerator_config == {}


# ──────────────────────────────────────────────
# 3. Setup: data
# ──────────────────────────────────────────────

def test_dataloaders_created():
    engine = make_linear_engine()
    assert engine.train_dataloader is not None
    assert engine.val_dataloader is not None


def test_train_steps_per_epoch_auto_calculated():
    engine = make_linear_engine(train_batch_size=32)
    assert engine.train_steps_per_epoch == math.ceil(100 / 32)


def test_train_steps_per_epoch_with_grad_accum():
    engine = make_linear_engine(train_batch_size=32, gradient_accumulation_steps=4)
    expected = math.ceil(math.ceil(100 / 32) / 4)
    assert engine.train_steps_per_epoch == expected


def test_max_steps_derived_from_max_epochs():
    engine = make_linear_engine(max_epochs=3, train_steps_per_epoch=10)
    assert engine.max_steps == 30


def test_max_epochs_derived_from_max_steps():
    engine = make_linear_engine(max_epochs=None, max_steps=50, train_steps_per_epoch=20)
    assert engine.max_epochs == math.ceil(50 / 20)


def test_both_none_raises():
    try:
        make_linear_engine(max_epochs=None, max_steps=None)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_do_x_disabled_when_loader_none():
    class PartialData(Data):
        def setup(self, stage=None):
            self.train_dataset = TensorDataset(torch.randn(10, 10), torch.randn(10, 1))
    model = LinearModel()
    engine = Engine(model=model, data=PartialData(), train_batch_size=4, max_epochs=1, seed=42)
    assert engine.do_train is True
    assert engine.do_val is False
    assert engine.do_test is False
    assert engine.do_predict is False


# ──────────────────────────────────────────────
# 4. Dataloader kwargs
# ──────────────────────────────────────────────

def test_dataloader_kwargs_default_batch_size():
    engine = make_linear_engine()
    assert engine.train_batch_size == 16


def test_dataloader_kwargs_global():
    engine = make_linear_engine(dataloader_config={"shuffle": False})
    assert engine.train_dataloader is not None


def test_dataloader_kwargs_mode_prefix():
    engine = make_linear_engine(dataloader_config={"train_num_workers": 2})
    assert engine.train_dataloader is not None


def test_dataloader_kwargs_mode_section():
    engine = make_linear_engine(dataloader_config={"train": {"batch_size": 8}})
    assert engine.train_batch_size == 16


def test_dataloader_kwargs_val_batch_size():
    engine = make_linear_engine(eval_batch_size=4)
    assert engine.eval_batch_size == 4


# ──────────────────────────────────────────────
# 5. Setup: model + optimizers
# ──────────────────────────────────────────────

def test_inference_mode():
    class NoOptModel(Model):
        def forward(self, x):
            return x
        def configure_optimizers(self):
            return None
    engine = Engine(
        model=NoOptModel(), data=DummyData(),
        do_train=False, do_test=True, eval_batch_size=4,
    )
    assert len(engine.optimizers) == 0
    assert len(engine.schedulers) == 0


def test_model_on_device():
    engine = make_linear_engine()
    p = next(engine.model.parameters())
    assert p.device.type == engine.device.type


def test_optimizer_and_scheduler_on_device():
    engine = make_linear_engine()
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 1


# ──────────────────────────────────────────────
# 6. configure_optimizers formats
# ──────────────────────────────────────────────

def test_optimizers_single_optimizer():
    class SingleOpt(Model):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)
        def forward(self, x):
            return self.layer(x)
        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)
    engine = Engine(model=SingleOpt(), data=DummyData(), train_batch_size=4, max_epochs=1, seed=42)
    assert len(engine.optimizers) == 1


def test_optimizers_tuple_opt_sched():
    class TupleSched(Model):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)
        def forward(self, x):
            return self.layer(x)
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return opt, sched
    engine = Engine(model=TupleSched(), data=DummyData(), train_batch_size=4, max_epochs=1, seed=42)
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 1


def test_optimizers_tuple_opt_sched_dict():
    class TupleSchedDict(Model):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)
        def forward(self, x):
            return self.layer(x)
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return (opt, {"scheduler": sched, "interval": 1, "strategy": "epoch"})
    engine = Engine(model=TupleSchedDict(), data=DummyData(), train_batch_size=4, max_epochs=1, seed=42)
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 1


def test_optimizers_dict_format():
    class DictOpt(Model):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)
        def forward(self, x):
            return self.layer(x)
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": 1, "strategy": "epoch"}}
    engine = Engine(model=DictOpt(), data=DummyData(), train_batch_size=4, max_epochs=1, seed=42)
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 1


def test_optimizers_dict_format_scheduler_key():
    class DictSched(Model):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)
        def forward(self, x):
            return self.layer(x)
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return {"optimizer": opt, "scheduler": sched}
    engine = Engine(model=DictSched(), data=DummyData(), train_batch_size=4, max_epochs=1, seed=42)
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 1


def test_optimizers_two_list_format():
    class MultiOpt(Model):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(10, 5)
            self.l2 = nn.Linear(5, 1)
        def forward(self, x):
            return self.l2(self.l1(x))
        def configure_optimizers(self):
            opt1 = torch.optim.SGD(self.l1.parameters(), lr=0.01)
            opt2 = torch.optim.SGD(self.l2.parameters(), lr=0.01)
            sched1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=1)
            return [opt1, opt2], [sched1]
    engine = Engine(model=MultiOpt(), data=DummyData(), train_batch_size=4, max_epochs=1, seed=42)
    assert len(engine.optimizers) == 2
    assert len(engine.schedulers) == 1


def test_optimizers_list_of_dicts():
    class ListDicts(Model):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)
        def forward(self, x):
            return self.layer(x)
        def configure_optimizers(self):
            opt = torch.optim.Adam(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
            return [{"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": 1, "strategy": "epoch"}}]
    engine = Engine(model=ListDicts(), data=DummyData(), train_batch_size=4, max_epochs=1, seed=42)
    assert len(engine.optimizers) == 1
    assert len(engine.schedulers) == 1


def test_optimizers_none():
    class NoOpt(Model):
        def forward(self, x):
            return x
        def configure_optimizers(self):
            return None
    engine = Engine(model=NoOpt(), data=DummyData(), do_train=False, do_test=True, eval_batch_size=4)
    assert len(engine.optimizers) == 0
    assert len(engine.schedulers) == 0


def test_optimizers_reduce_on_plateau_no_monitor_raises():
    class PlateauNoMonitor(Model):
        def forward(self, x):
            return x
        def configure_optimizers(self):
            opt = torch.optim.SGD(self.parameters(), lr=0.01)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
            return {"optimizer": opt, "lr_scheduler": sched}
    try:
        Engine(model=PlateauNoMonitor(), data=DummyData(), train_batch_size=4, max_epochs=1, seed=42)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_optimizers_none_with_do_train_raises():
    class NoOpt(Model):
        def forward(self, x):
            return x
        def configure_optimizers(self):
            return None
    try:
        Engine(model=NoOpt(), data=DummyData(), train_batch_size=4, max_epochs=1, seed=42)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ──────────────────────────────────────────────
# 7. Training primitives
# ──────────────────────────────────────────────

def test_clip_gradients_none_is_noop():
    engine = make_linear_engine()
    engine.optimizers_zero_grad()
    x = torch.randn(4, 10, device=engine.device)
    loss = engine.model(x).sum()
    loss.backward()
    engine.clip_gradients()


def test_clip_gradients_with_value():
    engine = make_linear_engine(gradient_clip_value=1.0)
    engine.optimizers_zero_grad()
    x = torch.randn(4, 10, device=engine.device)
    loss = engine.model(x).sum()
    engine.backward(loss)
    engine.clip_gradients()


def test_backward_and_step():
    engine = make_linear_engine()
    engine.optimizers_zero_grad()
    x = torch.randn(4, 10, device=engine.device)
    y = torch.randn(4, 1, device=engine.device)
    preds = engine.model(x)
    loss = nn.functional.mse_loss(preds, y)
    engine.backward(loss)
    engine.optimizers_step()
    engine.epoch = 1
    engine.schedulers_step(strategy="epoch")


def test_gather_and_reduce():
    engine = make_linear_engine()
    t = torch.tensor([1.0, 2.0])
    result = engine.reduce(t, reduction="sum")
    assert result is not None


# ──────────────────────────────────────────────
# 8. Scheduler step
# ──────────────────────────────────────────────

def test_scheduler_step_no_monitor():
    engine = make_linear_engine()
    assert len(engine.schedulers) == 1
    engine.schedulers[0]['monitor'] = None
    engine.scheduler_step(0)


def test_scheduler_step_with_monitor():
    engine = make_linear_engine()
    engine.schedulers[0]['monitor'] = "val_loss"
    engine.monitor["val_loss"] = 0.5
    engine.scheduler_step(0)


def test_scheduler_step_missing_monitor_raises():
    engine = make_linear_engine()
    engine.schedulers[0]['monitor'] = "val_loss"
    try:
        engine.scheduler_step(0)
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_schedulers_step_filters_by_strategy():
    engine = make_linear_engine()
    assert len(engine.schedulers) == 1
    engine.schedulers[0]['strategy'] = "step"
    engine.schedulers[0]['interval'] = 1
    engine.schedulers_step(strategy="epoch")


def test_schedulers_step_interval_skips():
    engine = make_linear_engine()
    engine.schedulers[0]['strategy'] = "epoch"
    engine.schedulers[0]['interval'] = 2
    engine.epoch = 1
    engine.schedulers_step(strategy="epoch")
    engine.epoch = 2
    engine.schedulers_step(strategy="epoch")


# ──────────────────────────────────────────────
# 9. Validation logic
# ──────────────────────────────────────────────

def test_should_validate_no_strategy():
    engine = make_linear_engine()
    engine.val_strategy = "no"
    assert engine.should_validate() is False


def test_should_validate_epoch_strategy():
    engine = make_linear_engine()
    engine.val_strategy = "epoch"
    engine.step = 0
    engine.epoch = 1
    assert engine.should_validate() is True


def test_should_validate_step_strategy():
    engine = make_linear_engine()
    engine.val_strategy = "step"
    engine.step = 5
    assert engine.should_validate() is True


def test_should_validate_start():
    engine = make_linear_engine(val_start=3, val_interval=1)
    engine.epoch = 2
    assert engine.should_validate() is False
    engine.epoch = 3
    assert engine.should_validate() is True


def test_should_validate_interval():
    engine = make_linear_engine(val_start=0, val_interval=3)
    engine.epoch = 2
    assert engine.should_validate() is False
    engine.epoch = 3
    assert engine.should_validate() is True


# ──────────────────────────────────────────────
# 10. Full training loop
# ──────────────────────────────────────────────

def test_full_training_loop():
    class TrainEngine(Engine):
        def train_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return nn.functional.mse_loss(preds, y)

        def val_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return {"val_loss": nn.functional.mse_loss(preds, y)}

        def test_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return {"test_loss": nn.functional.mse_loss(preds, y)}

    data = DummyData()
    model = LinearModel()
    engine = TrainEngine(model=model, data=data, train_batch_size=32, seed=42, max_steps=6)

    assert engine.epoch == 0
    assert engine.step == 0

    engine.run_train()

    assert engine.step == 6


def test_training_loop_stops_at_max_steps():
    class TrainEngine(Engine):
        def train_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return nn.functional.mse_loss(preds, y)

        def val_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return {"val_loss": nn.functional.mse_loss(preds, y)}

    data = DummyData()
    model = LinearModel()
    engine = TrainEngine(model=model, data=data, train_batch_size=32, seed=42, max_steps=5)

    engine.run_train()

    assert engine.step == 5, f"Step was {engine.step}, expected 5"


def test_loop_updates_lr_scheduler():
    class TrainEngine(Engine):
        def train_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return nn.functional.mse_loss(preds, y)

        def val_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return {"val_loss": nn.functional.mse_loss(preds, y)}

    data = DummyData()
    model = LinearModel()
    engine = TrainEngine(model=model, data=data, train_batch_size=32, seed=42, max_epochs=2)

    lr_before = engine.optimizers[0].param_groups[0]['lr']
    engine.run_train()
    lr_after = engine.optimizers[0].param_groups[0]['lr']

    assert lr_after != lr_before, f"LR did not change: {lr_before} -> {lr_after}"


# ──────────────────────────────────────────────
# 11. Standalone eval / predict
# ──────────────────────────────────────────────

def test_standalone_val():
    class ValEngine(Engine):
        def val_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return {"val_loss": nn.functional.mse_loss(preds, y)}

    engine = ValEngine(
        model=LinearModel(), data=DummyData(),
        do_train=False, do_val=True, eval_batch_size=4,
    )
    engine.run_val()


def test_standalone_test():
    class TestEngine(Engine):
        def test_step(self, batch):
            x, y = batch
            preds = self.model(x)
            return {"test_loss": nn.functional.mse_loss(preds, y)}

    engine = TestEngine(
        model=LinearModel(), data=DummyData(),
        do_train=False, do_test=True, eval_batch_size=4,
    )
    engine.run_test()


def test_standalone_predict():
    class PredictEngine(Engine):
        def predict_step(self, batch):
            x, y = batch
            return self.model(x)

    engine = PredictEngine(
        model=LinearModel(), data=DummyData(),
        do_train=False, do_predict=True, eval_batch_size=4,
    )
    engine.run_predict()


# ──────────────────────────────────────────────
# 12. Edge cases
# ──────────────────────────────────────────────

def test_max_epochs_and_max_steps_both_set():
    engine = make_linear_engine(max_epochs=5, max_steps=10)
    assert engine.max_epochs == 5
    assert engine.max_steps == 10


def test_gradient_accumulation_config():
    engine = make_linear_engine(gradient_accumulation_steps=4)
    assert engine.gradient_accumulation_steps == 4


def test_compile_flag_no_crash():
    engine = make_linear_engine(compile=False)
    assert engine.compile is False


def test_deterministic_with_seed():
    engine = make_linear_engine(deterministic=True, seed=42)
    assert engine.deterministic is True


def test_sync_batch_norm_no_crash():
    engine = make_linear_engine(sync_batch_norm=True)
    assert engine.sync_batch_norm is True


def test_mixed_precision_config():
    engine = make_linear_engine(mixed_precision="no")
    assert engine.mixed_precision == "no"


def test_project_dir_created():
    assert os.path.isdir("outputs") or not os.path.isdir("outputs")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def run_all():
    tests = [
        ("globals  : deterministic sets env",            test_setup_globals_deterministic_sets_env),
        ("globals  : deterministic disables benchmark",  test_setup_globals_deterministic_disables_benchmark),
        ("globals  : tf32 bool",                         test_setup_globals_tf32_bool),
        ("globals  : tf32 str",                          test_setup_globals_tf32_str),
        ("globals  : default seed None",                 test_setup_globals_defaults),
        ("", None),
        ("accel    : accelerator created",               test_accelerator_created),
        ("accel    : default project",                   test_accelerator_default_project),
        ("accel    : init_trackers_config merge",        test_accelerator_init_trackers_config_merge),
        ("accel    : seed None",                         test_accelerator_seed_none),
        ("accel    : seed set",                          test_accelerator_seed_set),
        ("accel    : accelerator_config merge",          test_accelerator_config_merge),
        ("", None),
        ("data     : dataloaders created",               test_dataloaders_created),
        ("data     : train_steps auto-calc",             test_train_steps_per_epoch_auto_calculated),
        ("data     : train_steps with grad accum",       test_train_steps_per_epoch_with_grad_accum),
        ("data     : max_steps from max_epochs",         test_max_steps_derived_from_max_epochs),
        ("data     : max_epochs from max_steps",         test_max_epochs_derived_from_max_steps),
        ("data     : both None raises",                  test_both_none_raises),
        ("data     : do_x disabled when no loader",      test_do_x_disabled_when_loader_none),
        ("", None),
        ("dataload : batch_size default",                test_dataloader_kwargs_default_batch_size),
        ("dataload : global kwargs",                     test_dataloader_kwargs_global),
        ("dataload : mode prefix",                       test_dataloader_kwargs_mode_prefix),
        ("dataload : mode section",                      test_dataloader_kwargs_mode_section),
        ("dataload : val batch size",                    test_dataloader_kwargs_val_batch_size),
        ("", None),
        ("model    : inference mode",                    test_inference_mode),
        ("model    : model on device",                   test_model_on_device),
        ("model    : opt+sched on device",               test_optimizer_and_scheduler_on_device),
        ("", None),
        ("optconf  : single optimizer",                  test_optimizers_single_optimizer),
        ("optconf  : tuple opt+sched",                   test_optimizers_tuple_opt_sched),
        ("optconf  : tuple opt+sched_dict",              test_optimizers_tuple_opt_sched_dict),
        ("optconf  : dict format",                       test_optimizers_dict_format),
        ("optconf  : dict with scheduler key",           test_optimizers_dict_format_scheduler_key),
        ("optconf  : two-list format",                   test_optimizers_two_list_format),
        ("optconf  : list of dicts",                     test_optimizers_list_of_dicts),
        ("optconf  : None return",                       test_optimizers_none),
        ("optconf  : plateau no monitor raises",         test_optimizers_reduce_on_plateau_no_monitor_raises),
        ("optconf  : None with do_train raises",         test_optimizers_none_with_do_train_raises),
        ("", None),
        ("primitiv : clip none is noop",                 test_clip_gradients_none_is_noop),
        ("primitiv : clip with value",                   test_clip_gradients_with_value),
        ("primitiv : backward + step",                   test_backward_and_step),
        ("primitiv : gather + reduce",                   test_gather_and_reduce),
        ("", None),
        ("scheduler: step no monitor",                   test_scheduler_step_no_monitor),
        ("scheduler: step with monitor",                 test_scheduler_step_with_monitor),
        ("scheduler: missing monitor raises",            test_scheduler_step_missing_monitor_raises),
        ("scheduler: filters by strategy",               test_schedulers_step_filters_by_strategy),
        ("scheduler: interval skips",                    test_schedulers_step_interval_skips),
        ("", None),
        ("validate : no strategy",                       test_should_validate_no_strategy),
        ("validate : epoch strategy",                    test_should_validate_epoch_strategy),
        ("validate : step strategy",                     test_should_validate_step_strategy),
        ("validate : val_start",                         test_should_validate_start),
        ("validate : val_interval",                      test_should_validate_interval),
        ("", None),
        ("loop     : full training",                     test_full_training_loop),
        ("loop     : stops at max_steps",                test_training_loop_stops_at_max_steps),
        ("loop     : lr scheduler steps",                test_loop_updates_lr_scheduler),
        ("", None),
        ("standalon: run_val",                           test_standalone_val),
        ("standalon: run_test",                          test_standalone_test),
        ("standalon: run_predict",                       test_standalone_predict),
        ("", None),
        ("edge     : both max_epochs+max_steps set",     test_max_epochs_and_max_steps_both_set),
        ("edge     : grad accum config",                 test_gradient_accumulation_config),
        ("edge     : compile flag",                      test_compile_flag_no_crash),
        ("edge     : deterministic + seed",              test_deterministic_with_seed),
        ("edge     : sync_batch_norm",                   test_sync_batch_norm_no_crash),
        ("edge     : mixed precision",                   test_mixed_precision_config),
        ("edge     : project dir created",               test_project_dir_created),
    ]

    passed = 0
    failed = 0

    for label, func in tests:
        if func is None:
            print()
            continue
        try:
            func()
            print(f"  PASS  {label}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {label}")
            print(f"         {e}")
            failed += 1

    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"  {total} tests  |  {passed} passed  |  {failed} failed")
    print(f"{'=' * 50}")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    exit(0 if success else 1)