from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.anima_entry_test_utils import import_native_entry


class _DummyLRScheduler:
    def get_last_lr(self):
        return [1.0e-4]


class _DummyAccelerator:
    def __init__(self, device: torch.device):
        self.device = device
        self.trackers = []

    def get_tracker(self, name):
        raise KeyError(name)


def _build_log_args(loss_spike_ratio: float = 3.0):
    return SimpleNamespace(
        network_train_unet_only=True,
        optimizer_type="AdamW",
        anima_monitor_loss_spike_ratio=loss_spike_ratio,
    )


def _build_runtime_args(*, monitor_memory: bool, alert_policy: str = "warn", warn_ratio: float = 0.95):
    return SimpleNamespace(
        anima_monitor_memory=monitor_memory,
        anima_monitor_alert_policy=alert_policy,
        anima_monitor_memory_warn_ratio=warn_ratio,
    )


def test_generate_step_logs_marks_nonfinite_loss():
    native_entry = import_native_entry()
    trainer = native_entry.AnimaNetworkTrainer()
    logs = trainer.generate_step_logs(
        _build_log_args(),
        float("nan"),
        1.0,
        _DummyLRScheduler(),
        None,
    )
    assert logs["alert/nonfinite_loss"] == 1.0
    assert logs["alert/any"] == 1.0


def test_generate_step_logs_marks_loss_spike():
    native_entry = import_native_entry()
    trainer = native_entry.AnimaNetworkTrainer()
    args = _build_log_args(loss_spike_ratio=2.0)
    trainer.generate_step_logs(args, 1.0, 1.0, _DummyLRScheduler(), None)
    logs = trainer.generate_step_logs(args, 3.1, 2.05, _DummyLRScheduler(), None)
    assert logs["alert/loss_spike"] == 1.0
    assert logs["alert/any"] == 1.0


def test_step_logging_raise_policy_raises_on_nonfinite():
    native_entry = import_native_entry()
    trainer = native_entry.AnimaNetworkTrainer()
    trainer._runtime_args = _build_runtime_args(monitor_memory=False, alert_policy="raise")

    accelerator = _DummyAccelerator(device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="nonfinite_loss"):
        trainer.step_logging(
            accelerator,
            {"alert/nonfinite_loss": 1.0, "alert/loss_spike": 0.0, "alert/memory_near_limit": 0.0},
            global_step=1,
            epoch=1,
        )


def test_collect_memory_monitor_logs_reports_metrics(monkeypatch):
    native_entry = import_native_entry()
    trainer = native_entry.AnimaNetworkTrainer()
    trainer._runtime_args = _build_runtime_args(monitor_memory=True, warn_ratio=0.5)
    accelerator = _DummyAccelerator(device=torch.device("cuda", 0))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _idx: 128 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _idx: 700 * 1024 * 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _idx: 900 * 1024 * 1024)

    class _Props:
        total_memory = 1000 * 1024 * 1024

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _idx: _Props())

    logs = trainer._collect_memory_monitor_logs(accelerator)
    assert logs["gpu/mem_allocated_mb"] == pytest.approx(128.0)
    assert logs["gpu/mem_reserved_mb"] == pytest.approx(700.0)
    assert logs["gpu/mem_peak_allocated_mb"] == pytest.approx(900.0)
    assert logs["alert/memory_near_limit"] == 1.0
