from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

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
        self.norm = torch.nn.LayerNorm(8)


class _LlmAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_LlmAdapterBlock()])
        self.norm = _CustomRMSNorm(8)


class _DummyAnimaTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_LlmAdapterBlock(), _LlmAdapterBlock()])
        self.llm_adapter = _LlmAdapter()

    def forward(self, x):
        block = self.blocks[0]
        x = block.self_attn.q_proj(block.self_attn.q_norm(x))
        x = block.mlp[2](block.mlp[1](block.mlp[0](x)))
        x = self.llm_adapter.blocks[0].cross_attn.q_proj(x)
        return self.llm_adapter.norm(x)


def _export_weights(network_module, tmp_path: Path, stem: str):
    unet = _DummyAnimaTransformer()
    network = network_module.create_network(
        1.0,
        4,
        4.0,
        None,
        None,
        unet,
        train_norm="true",
    )
    network.apply_to(None, unet, False, True)

    # Ensure both adapter and norm deltas are non-zero for export assertions.
    with torch.no_grad():
        for param in network.get_trainable_params():
            param.add_(0.01)

    ckpt = tmp_path / f"{stem}.safetensors"
    network.save_weights(str(ckpt), torch.float32, metadata={"test": "1"})
    return load_file(str(ckpt))


def test_lokr_export_uses_comfy_keys_and_includes_norm(tmp_path: Path):
    sd = _export_weights(lokr_anima, tmp_path, "lokr")
    keys = list(sd.keys())

    assert keys
    assert all(key.startswith("diffusion_model.") for key in keys)
    assert any(key.endswith(".lokr_w1") for key in keys)
    assert any(key.endswith(".lokr_w2_a") or key.endswith(".lokr_w2") for key in keys)
    assert any(key.endswith(".alpha") for key in keys)
    assert any(key.endswith(".w_norm") for key in keys)
    assert any(key.endswith(".b_norm") for key in keys)
    assert any("llm_adapter" in key for key in keys)


def test_lora_export_uses_comfy_keys_and_includes_norm(tmp_path: Path):
    sd = _export_weights(lora_anima, tmp_path, "lora")
    keys = list(sd.keys())

    assert keys
    assert all(key.startswith("diffusion_model.") for key in keys)
    assert any(key.endswith(".lora_down.weight") for key in keys)
    assert any(key.endswith(".lora_up.weight") for key in keys)
    assert any(key.endswith(".alpha") for key in keys)
    assert any(key.endswith(".w_norm") for key in keys)
    assert any(key.endswith(".b_norm") for key in keys)
    assert any("llm_adapter" in key for key in keys)
