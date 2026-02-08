from __future__ import annotations

import math
import os
import fnmatch
import re
from typing import Dict, List, Optional

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


DEFAULT_TARGETS_V101 = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.k_proj",
    "cross_attn.v_proj",
    "cross_attn.o_proj",
    "cross_attn.output_proj",
    "mlp.layer1",
    "mlp.layer2",
    "mlp.0",
    "mlp.2",
]


def _factorization(dimension: int, factor: int) -> tuple[int, int]:
    dimension = max(1, int(dimension))
    factor = int(factor)
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n

    if factor < 0:
        factor = dimension

    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n

    if m > n:
        n, m = m, n
    return m, n


class _LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_down = torch.nn.Linear(in_features, self.rank, bias=False)
        self.lora_up = torch.nn.Linear(self.rank, out_features, bias=False)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        torch.nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


class _LoKrLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        alpha=1.0,
        factor=8,
        full_matrix=False,
        decompose_both=False,
        rank_dropout=0.0,
        module_dropout=0.0,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.factor = max(1, int(factor))
        self.full_matrix = bool(full_matrix)
        self.decompose_both = bool(decompose_both)
        self.rank_dropout = float(rank_dropout)
        self.module_dropout = float(module_dropout)

        self.out_l, self.out_k = _factorization(self.out_features, self.factor)
        self.in_m, self.in_n = _factorization(self.in_features, self.factor)

        w1_threshold = max(self.out_l, self.in_m) / 2.0
        w2_threshold = max(self.out_k, self.in_n) / 2.0
        self.w1_direct = self.full_matrix or (not (self.decompose_both and self.rank < w1_threshold))
        self.w2_direct = self.full_matrix or (self.rank >= w2_threshold)

        effective_alpha = self.alpha if self.alpha != 0 else float(self.rank)
        if self.w1_direct and self.w2_direct:
            effective_alpha = float(self.rank)
        self.scaling = effective_alpha / float(self.rank)

        if self.w1_direct:
            self.lokr_w1 = torch.nn.Parameter(torch.empty(self.out_l, self.in_m))
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=5**0.5)
        else:
            self.lokr_w1_a = torch.nn.Parameter(torch.empty(self.out_l, self.rank))
            self.lokr_w1_b = torch.nn.Parameter(torch.empty(self.rank, self.in_m))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=5**0.5)
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=5**0.5)

        if self.w2_direct:
            self.lokr_w2 = torch.nn.Parameter(torch.zeros(self.out_k, self.in_n))
        else:
            self.lokr_w2_a = torch.nn.Parameter(torch.empty(self.out_k, self.rank))
            self.lokr_w2_b = torch.nn.Parameter(torch.zeros(self.rank, self.in_n))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=5**0.5)

    def _w1(self):
        if self.w1_direct:
            return self.lokr_w1
        return self.lokr_w1_a @ self.lokr_w1_b

    def _w2(self):
        if self.w2_direct:
            return self.lokr_w2
        return self.lokr_w2_a @ self.lokr_w2_b

    def forward(self, x):
        if self.training and self.module_dropout > 0 and torch.rand(()) < self.module_dropout:
            return torch.zeros((*x.shape[:-1], self.out_features), device=x.device, dtype=x.dtype)
        diff = torch.kron(self._w1(), self._w2()).reshape(self.out_features, self.in_features) * self.scaling
        diff = diff.to(device=x.device, dtype=x.dtype)
        if self.training and self.rank_dropout > 0:
            mask = (torch.rand(diff.shape[0], device=diff.device) > self.rank_dropout).to(diff.dtype)
            diff = diff * mask[:, None]
        return F.linear(x, diff)


class _LoRAInjectedLinear(torch.nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.original = original_layer
        self.adapter = _LoRALayer(original_layer.in_features, original_layer.out_features, rank=rank, alpha=alpha, dropout=dropout)
        self.multiplier = 1.0
        self.adapter.to(device=original_layer.weight.device, dtype=original_layer.weight.dtype)
        for p in self.original.parameters():
            p.requires_grad = False

    def set_multiplier(self, value: float):
        self.multiplier = float(value)

    def forward(self, x):
        return self.original(x) + self.multiplier * self.adapter(x)


class _LoKrInjectedLinear(torch.nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1.0, dropout=0.0, factor=8, full_matrix=False, decompose_both=False, rank_dropout=0.0, module_dropout=0.0):
        super().__init__()
        self.original = original_layer
        self.input_dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.adapter = _LoKrLayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha,
            factor=factor,
            full_matrix=full_matrix,
            decompose_both=decompose_both,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )
        self.multiplier = 1.0
        self.adapter.to(device=original_layer.weight.device, dtype=original_layer.weight.dtype)
        for p in self.original.parameters():
            p.requires_grad = False

    def set_multiplier(self, value: float):
        self.multiplier = float(value)

    def forward(self, x):
        return self.original(x) + self.multiplier * self.adapter(self.input_dropout(x))


class AnimaAdapterNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        network_type: str,
        multiplier: float = 1.0,
        network_dim: int = 4,
        network_alpha: float = 1.0,
        neuron_dropout: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
        lokr_factor: int = 8,
        lokr_full_matrix: bool = False,
        lokr_decompose_both: bool = False,
        lokr_rank_dropout: float = 0.0,
        lokr_module_dropout: float = 0.0,
    ):
        super().__init__()
        self.network_type = str(network_type).lower()
        self.multiplier = float(multiplier)
        self.network_dim = max(1, int(network_dim))
        self.network_alpha = float(network_alpha)
        self.neuron_dropout = float(neuron_dropout or 0.0)
        self.target_modules = [t.strip() for t in (target_modules or DEFAULT_TARGETS_V101) if str(t).strip()]
        self.lokr_factor = int(lokr_factor)
        self.lokr_full_matrix = bool(lokr_full_matrix)
        self.lokr_decompose_both = bool(lokr_decompose_both)
        self.lokr_rank_dropout = float(lokr_rank_dropout)
        self.lokr_module_dropout = float(lokr_module_dropout)
        self.injected_layers: Dict[str, torch.nn.Module] = {}
        self._trainable_params: List[torch.nn.Parameter] = []
        self._last_weight_norms: Optional[torch.Tensor] = None
        self._last_grad_norms: Optional[torch.Tensor] = None

    def _match_target(self, module_name: str) -> bool:
        leaf = module_name.split(".")[-1]
        for target in self.target_modules:
            if target.startswith("re:"):
                if re.search(target[3:], module_name):
                    return True
                continue
            if target.startswith("transformer."):
                target = target[len("transformer.") :]
            if "*" in target or "?" in target:
                if fnmatch.fnmatch(module_name, target):
                    return True
                continue
            if "." in target:
                if module_name == target or module_name.endswith("." + target):
                    return True
                continue
            if leaf == target:
                return True
        return False

    def _wrap_linear(self, module: torch.nn.Linear):
        if self.network_type == "lokr":
            wrapped = _LoKrInjectedLinear(
                module,
                rank=self.network_dim,
                alpha=self.network_alpha,
                dropout=self.neuron_dropout,
                factor=self.lokr_factor,
                full_matrix=self.lokr_full_matrix,
                decompose_both=self.lokr_decompose_both,
                rank_dropout=self.lokr_rank_dropout,
                module_dropout=self.lokr_module_dropout,
            )
        else:
            wrapped = _LoRAInjectedLinear(
                module,
                rank=self.network_dim,
                alpha=self.network_alpha,
                dropout=self.neuron_dropout,
            )
        wrapped.set_multiplier(self.multiplier)
        return wrapped

    def apply_to(self, text_encoder, unet, train_text_encoder=True, train_unet=True):
        modules = list(unet.named_modules())
        injected = {}
        for name, module in modules:
            if not isinstance(module, torch.nn.Linear):
                continue
            if not self._match_target(name):
                continue
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = unet.get_submodule(parent_name) if parent_name else unet
            wrapped = self._wrap_linear(module)
            setattr(parent, child_name, wrapped)
            injected[name] = wrapped

        self.injected_layers = injected
        params: List[torch.nn.Parameter] = []
        for wrapped in injected.values():
            params.extend([p for p in wrapped.adapter.parameters() if p.requires_grad])
        self._trainable_params = params
        logger.info("Anima %s injected layers: %d", self.network_type.upper(), len(self.injected_layers))

    def set_multiplier(self, multiplier: float):
        self.multiplier = float(multiplier)
        for layer in self.injected_layers.values():
            if hasattr(layer, "set_multiplier"):
                layer.set_multiplier(self.multiplier)

    def get_trainable_params(self):
        return self._trainable_params

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None):
        lr = unet_lr if unet_lr is not None else default_lr
        if lr is None:
            return [{"params": self.get_trainable_params()}]
        return [{"params": self.get_trainable_params(), "lr": lr}]

    def prepare_network(self, args):
        return None

    def enable_gradient_checkpointing(self):
        return None

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def _state_pairs(self, name: str, layer: torch.nn.Module):
        if self.network_type == "lokr":
            adapter = layer.adapter
            pairs = []
            if hasattr(adapter, "lokr_w1"):
                pairs.append((f"{name}.lokr_w1", adapter.lokr_w1))
            if hasattr(adapter, "lokr_w2"):
                pairs.append((f"{name}.lokr_w2", adapter.lokr_w2))
            if hasattr(adapter, "lokr_w1_a"):
                pairs.append((f"{name}.lokr_w1_a", adapter.lokr_w1_a))
            if hasattr(adapter, "lokr_w1_b"):
                pairs.append((f"{name}.lokr_w1_b", adapter.lokr_w1_b))
            if hasattr(adapter, "lokr_w2_a"):
                pairs.append((f"{name}.lokr_w2_a", adapter.lokr_w2_a))
            if hasattr(adapter, "lokr_w2_b"):
                pairs.append((f"{name}.lokr_w2_b", adapter.lokr_w2_b))
            return pairs
        return [
            (f"{name}.lora_down.weight", layer.adapter.lora_down.weight),
            (f"{name}.lora_up.weight", layer.adapter.lora_up.weight),
        ]

    def _adapter_state_dict(self):
        sd = {}
        for name, layer in self.injected_layers.items():
            for key, tensor in self._state_pairs(name, layer):
                sd[key] = tensor.detach().cpu()
        return sd

    def _load_adapter_state_dict(self, weights_sd: Dict[str, torch.Tensor], strict: bool = True):
        expected = {}
        for name, layer in self.injected_layers.items():
            for key, tensor in self._state_pairs(name, layer):
                expected[key] = tensor

        missing = [k for k in expected if k not in weights_sd]
        unexpected = [k for k in weights_sd if k not in expected]
        if strict and missing:
            raise RuntimeError(f"Missing adapter keys: {missing[:3]}")
        if strict and unexpected:
            raise RuntimeError(f"Unexpected adapter keys: {unexpected[:3]}")

        for k, p in expected.items():
            if k not in weights_sd:
                continue
            p.data.copy_(weights_sd[k].to(device=p.device, dtype=p.dtype))
        return {"missing": missing, "unexpected": unexpected}

    def save_weights(self, file: str, dtype: Optional[torch.dtype], metadata: Optional[dict] = None):
        sd = self._adapter_state_dict()
        if dtype is not None:
            sd = {k: v.to(dtype=dtype) if torch.is_floating_point(v) else v for k, v in sd.items()}
        metadata = metadata or {}
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(sd, file, metadata={k: str(v) for k, v in metadata.items()})
        else:
            torch.save({"state_dict": sd, "metadata": metadata}, file)

    def load_weights(self, file: str):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            obj = torch.load(file, map_location="cpu")
            weights_sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        info = self._load_adapter_state_dict(weights_sd, strict=False)
        return f"loaded: missing={len(info['missing'])} unexpected={len(info['unexpected'])}"

    def merge_to(self, text_encoder, unet, weights_sd, dtype=None, device=None):
        if not self.injected_layers:
            self.apply_to(text_encoder, unet, train_text_encoder=False, train_unet=True)
        self._load_adapter_state_dict(weights_sd, strict=False)

    def apply_max_norm_regularization(self, scale: float, device: torch.device):
        return 0, None, None

    def update_norms(self):
        if not self._trainable_params:
            self._last_weight_norms = None
            return
        vals = [p.detach().float().norm() for p in self._trainable_params]
        self._last_weight_norms = torch.stack(vals) if vals else None

    def update_grad_norms(self):
        vals = []
        for p in self._trainable_params:
            if p.grad is None:
                continue
            vals.append(p.grad.detach().float().norm())
        self._last_grad_norms = torch.stack(vals) if vals else None

    def weight_norms(self):
        return self._last_weight_norms

    def grad_norms(self):
        return self._last_grad_norms

    def combined_weight_norms(self):
        if self._last_weight_norms is None or self._last_grad_norms is None:
            return None
        n = min(self._last_weight_norms.numel(), self._last_grad_norms.numel())
        if n <= 0:
            return None
        return self._last_weight_norms[:n] + self._last_grad_norms[:n]


def _parse_target_modules(kwargs) -> List[str]:
    value = kwargs.get("target_modules", None)
    if value is None:
        return list(DEFAULT_TARGETS_V101)
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, (list, tuple)):
        return [str(t).strip() for t in value if str(t).strip()]
    return list(DEFAULT_TARGETS_V101)


def build_anima_network(network_type: str, multiplier: float, network_dim: Optional[int], network_alpha: Optional[float], **kwargs):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0
    target_modules = _parse_target_modules(kwargs)
    network = AnimaAdapterNetwork(
        network_type=network_type,
        multiplier=multiplier,
        network_dim=int(network_dim),
        network_alpha=float(network_alpha),
        neuron_dropout=float(kwargs.get("neuron_dropout", 0.0) or 0.0),
        target_modules=target_modules,
        lokr_factor=int(kwargs.get("lokr_factor", 8) or 8),
        lokr_full_matrix=bool(kwargs.get("lokr_full_matrix", False)),
        lokr_decompose_both=bool(kwargs.get("lokr_decompose_both", False)),
        lokr_rank_dropout=float(kwargs.get("lokr_rank_dropout", 0.0) or 0.0),
        lokr_module_dropout=float(kwargs.get("lokr_module_dropout", 0.0) or 0.0),
    )
    return network


def infer_network_dim_from_weights(network_type: str, weights_sd: Dict[str, torch.Tensor]) -> int:
    for key, value in weights_sd.items():
        if network_type == "lokr":
            if key.endswith(".lokr_w1_a") and value.ndim == 2:
                return int(value.shape[1])
            if key.endswith(".lokr_w2_a") and value.ndim == 2:
                return int(value.shape[1])
            if key.endswith(".lokr_w1") and value.ndim == 2:
                return int(min(value.shape))
        else:
            if key.endswith(".lora_down.weight") and value.ndim >= 2:
                return int(value.shape[0])
    return 4
