from __future__ import annotations

import logging

import torch

from library.anima_runtime import model_loading


class _FakeSafeOpen:
    def __init__(self, tensors):
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def keys(self):
        return list(self._tensors.keys())

    def get_tensor(self, key):
        return self._tensors[key]


def _fake_safe_open_factory(tensors):
    def _fake_safe_open(*args, **kwargs):
        del args, kwargs
        return _FakeSafeOpen(tensors)

    return _fake_safe_open


def _build_dummy_anima(load_result):
    class _DummyAnima:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict, strict=False):
            del state_dict, strict
            return load_result

        def to(self, device=None, dtype=None):
            self.device = device
            self.dtype = dtype
            return self

    return _DummyAnima


def test_load_state_dict_filters_comfy_quant_keys(monkeypatch, caplog):
    tensors = {
        "net.blocks.0.self_attn.q_proj.weight": torch.ones(2, 2),
        "net.blocks.0.self_attn.q_proj.comfy_quant": torch.ones(1),
        "diffusion_model.blocks.1.cross_attn.k_proj.weight": torch.ones(3, 3),
    }
    monkeypatch.setattr("safetensors.safe_open", _fake_safe_open_factory(tensors))

    caplog.set_level(logging.INFO, logger=model_loading.__name__)
    state_dict, stats = model_loading.load_state_dict("dummy.safetensors", return_filter_stats=True)

    assert "blocks.0.self_attn.q_proj.weight" in state_dict
    assert "blocks.1.cross_attn.k_proj.weight" in state_dict
    assert all(not key.endswith(".comfy_quant") for key in state_dict.keys())
    assert stats["filtered_count"] == 1
    assert stats["filtered_samples"] == ["blocks.0.self_attn.q_proj.comfy_quant"]
    assert any("Filtered 1 ignorable Anima state keys" in rec.message for rec in caplog.records)


def test_load_anima_model_only_ignorable_unexpected_logs_info(monkeypatch, caplog):
    monkeypatch.setattr(model_loading, "load_anima_class", lambda: _build_dummy_anima(([], ["x.y.comfy_quant"])))
    monkeypatch.setattr(model_loading, "load_config_from_ckpt", lambda _path: {"num_blocks": 1, "model_channels": 2048})
    monkeypatch.setattr(
        model_loading,
        "load_state_dict",
        lambda *_args, **_kwargs: ({}, {"filtered_count": 3, "filtered_samples": ["a.comfy_quant"]}),
    )

    caplog.set_level(logging.INFO, logger=model_loading.__name__)
    _model = model_loading.load_anima_model("dummy.safetensors", torch.device("cpu"), torch.float32)

    assert any("ignored quantization aux keys" in rec.message for rec in caplog.records)
    assert not any("Anima model state mismatch" in rec.message for rec in caplog.records)


def test_load_anima_model_real_unexpected_still_warns(monkeypatch, caplog):
    monkeypatch.setattr(model_loading, "load_anima_class", lambda: _build_dummy_anima(([], ["x.y.bad_key"])))
    monkeypatch.setattr(model_loading, "load_config_from_ckpt", lambda _path: {"num_blocks": 1, "model_channels": 2048})
    monkeypatch.setattr(
        model_loading,
        "load_state_dict",
        lambda *_args, **_kwargs: ({}, {"filtered_count": 0, "filtered_samples": []}),
    )

    caplog.set_level(logging.WARNING, logger=model_loading.__name__)
    _model = model_loading.load_anima_model("dummy.safetensors", torch.device("cpu"), torch.float32)

    assert any("Anima model state mismatch" in rec.message for rec in caplog.records)
