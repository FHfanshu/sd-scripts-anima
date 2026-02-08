from __future__ import annotations

from pathlib import Path

import torch

from networks import lora_anima, lokr_anima


class _AttnBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8)
        self.k_proj = torch.nn.Linear(8, 8)
        self.v_proj = torch.nn.Linear(8, 8)
        self.o_proj = torch.nn.Linear(8, 8)
        self.output_proj = torch.nn.Linear(8, 8)


class _MlpBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(8, 16)
        self.layer2 = torch.nn.Linear(16, 8)


class DummyAnimaTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _AttnBlock()
        self.cross_attn = _AttnBlock()
        self.mlp = _MlpBlock()

    def forward(self, x):
        x = self.self_attn.q_proj(x)
        x = self.cross_attn.q_proj(x)
        return self.mlp.layer2(self.mlp.layer1(x))


def _run_basic_network_roundtrip(network_module, tmp_path: Path):
    unet = DummyAnimaTransformer()
    network = network_module.create_network(1.0, 4, 4.0, None, None, unet)
    network.apply_to(None, unet, False, True)

    params = network.get_trainable_params()
    assert params, "expected trainable adapter params"

    output = unet(torch.randn(2, 8))
    assert output.shape == (2, 8)

    ckpt = tmp_path / "adapter.pt"
    network.save_weights(str(ckpt), torch.float32, metadata={"test": "1"})
    assert ckpt.exists()

    network2 = network_module.create_network(1.0, 4, 4.0, None, None, unet)
    network2.apply_to(None, unet, False, True)
    info = network2.load_weights(str(ckpt))
    assert "loaded" in info


def test_lora_anima_network_roundtrip(tmp_path: Path):
    _run_basic_network_roundtrip(lora_anima, tmp_path)


def test_lokr_anima_network_roundtrip(tmp_path: Path):
    _run_basic_network_roundtrip(lokr_anima, tmp_path)
