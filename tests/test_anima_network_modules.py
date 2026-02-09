from __future__ import annotations

from pathlib import Path

import torch

from networks import lora_anima, lokr_anima


class _CustomRMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


class _AttnBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.k_proj = torch.nn.Linear(8, 8)
        self.v_proj = torch.nn.Linear(8, 8)
        self.o_proj = torch.nn.Linear(8, 8)
        self.output_proj = torch.nn.Linear(8, 8)
        self.q_norm = torch.nn.LayerNorm(8)
        self.k_norm = _CustomRMSNorm(8)


class _MlpBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(8, 16)
        self.layer2 = torch.nn.Linear(16, 8)
        self.norm = torch.nn.LayerNorm(8)


class _LlmAdapterBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _AttnBlock()
        self.cross_attn = _AttnBlock()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.GELU(),
            torch.nn.Linear(16, 8),
        )
        self.norm = _CustomRMSNorm(8)


class _LlmAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_LlmAdapterBlock()])
        self.norm = torch.nn.LayerNorm(8)


class DummyAnimaTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _AttnBlock()
        self.cross_attn = _AttnBlock()
        self.mlp = _MlpBlock()
        self.llm_adapter = _LlmAdapter()

    def forward(self, x):
        x = self.self_attn.q_proj(self.self_attn.q_norm(x))
        x = self.cross_attn.q_proj(x)
        x = self.mlp.layer2(self.mlp.layer1(self.mlp.norm(x)))
        block = self.llm_adapter.blocks[0]
        x = block.self_attn.q_proj(block.self_attn.q_norm(x))
        x = block.mlp[2](block.mlp[1](block.mlp[0](x)))
        x = block.norm(x)
        return self.llm_adapter.norm(x)


def _run_basic_network_roundtrip(network_module, tmp_path: Path, train_norm: bool = True):
    unet = DummyAnimaTransformer()
    network = network_module.create_network(
        1.0,
        4,
        4.0,
        None,
        None,
        unet,
        train_norm="true" if train_norm else "false",
    )
    network.apply_to(None, unet, False, True)

    params = network.get_trainable_params()
    assert params, "expected trainable adapter params"
    norm_param_ids = {id(p) for n, p in unet.named_parameters() if "norm" in n}
    trainable_ids = {id(p) for p in params}
    if train_norm:
        assert norm_param_ids & trainable_ids, "expected norm params to be trainable when train_norm=true"
    else:
        assert not (norm_param_ids & trainable_ids), "norm params should not be in optimizer params when train_norm=false"

    output = unet(torch.randn(2, 8))
    assert output.shape == (2, 8)

    ckpt = tmp_path / "adapter.pt"
    network.save_weights(str(ckpt), torch.float32, metadata={"test": "1"})
    assert ckpt.exists()

    network2 = network_module.create_network(1.0, 4, 4.0, None, None, unet)
    network2.apply_to(None, unet, False, True)
    info = network2.load_weights(str(ckpt))
    assert "loaded" in info

    # Required by train_network.NetworkTrainer contract: params must recover after base model freeze.
    unet.requires_grad_(False)
    assert any(not p.requires_grad for p in params)
    network.prepare_grad_etc(None, unet)
    assert all(p.requires_grad for p in params)
    network.on_epoch_start(None, unet)
    assert all(p.requires_grad for p in params)
    max_norm = network.apply_max_norm_regularization(1.0, torch.device("cpu"))
    assert isinstance(max_norm, tuple)


def test_lora_anima_network_roundtrip(tmp_path: Path):
    _run_basic_network_roundtrip(lora_anima, tmp_path)


def test_lokr_anima_network_roundtrip(tmp_path: Path):
    _run_basic_network_roundtrip(lokr_anima, tmp_path)


def test_lora_anima_network_roundtrip_without_norm(tmp_path: Path):
    _run_basic_network_roundtrip(lora_anima, tmp_path, train_norm=False)


def test_lokr_network_dim_full_matrix_sentinel():
    unet = DummyAnimaTransformer()
    network = lokr_anima.create_network(
        1.0,
        100000,
        1.0,
        None,
        None,
        unet,
    )
    assert network.network_type == "lokr"
    assert network.lokr_full_matrix is True

    network_small = lokr_anima.create_network(
        1.0,
        16,
        16.0,
        None,
        None,
        unet,
    )
    assert network_small.lokr_full_matrix is False
