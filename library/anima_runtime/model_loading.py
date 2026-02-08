from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)

T5_EXPECTED_VOCAB_SIZE = 32128
T5_REQUIRED_FILES = ("config.json", "spiece.model", "tokenizer.json")


def load_anima_class():
    from library.anima_backend.modelling.anima_modeling import Anima

    return Anima


def load_wan_vae_class():
    from library.anima_backend.modelling.wan.vae2_1 import WanVAE_

    return WanVAE_


def load_config_from_ckpt(ckpt_path: str | Path):
    from safetensors import safe_open

    with safe_open(str(ckpt_path), framework="pt", device="cpu") as handle:
        key = None
        for item in handle.keys():
            if item.endswith("x_embedder.proj.1.weight"):
                key = item
                break
        if key is None:
            raise RuntimeError("Could not find x_embedder.proj.1.weight in checkpoint")
        weight = handle.get_tensor(key)

    concat_padding_mask = True
    in_channels = (weight.shape[1] // 4) - int(concat_padding_mask)
    model_channels = weight.shape[0]
    if model_channels == 2048:
        num_blocks = 28
        num_heads = 16
    elif model_channels == 5120:
        num_blocks = 36
        num_heads = 40
    else:
        raise RuntimeError(f"Unexpected model_channels={model_channels}")
    if in_channels == 16:
        rope_h_extrapolation_ratio = 4.0
        rope_w_extrapolation_ratio = 4.0
    elif in_channels == 17:
        rope_h_extrapolation_ratio = 3.0
        rope_w_extrapolation_ratio = 3.0
    else:
        rope_h_extrapolation_ratio = 1.0
        rope_w_extrapolation_ratio = 1.0

    return dict(
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        in_channels=in_channels,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=concat_padding_mask,
        model_channels=model_channels,
        num_blocks=num_blocks,
        num_heads=num_heads,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        min_fps=1,
        max_fps=30,
        use_adaln_lora=True,
        adaln_lora_dim=256,
        rope_h_extrapolation_ratio=rope_h_extrapolation_ratio,
        rope_w_extrapolation_ratio=rope_w_extrapolation_ratio,
        rope_t_extrapolation_ratio=1.0,
        extra_per_block_abs_pos_emb=False,
        extra_h_extrapolation_ratio=1.0,
        extra_w_extrapolation_ratio=1.0,
        extra_t_extrapolation_ratio=1.0,
        rope_enable_fps_modulation=False,
    )


def load_state_dict(path: str | Path):
    from safetensors import safe_open

    state_dict = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            mapped_key = key
            if mapped_key.startswith("net."):
                mapped_key = mapped_key[len("net.") :]
            if mapped_key.startswith("diffusion_model."):
                mapped_key = mapped_key[len("diffusion_model.") :]
            state_dict[mapped_key] = handle.get_tensor(key)
    return state_dict


def load_anima_model(ckpt_path, device, dtype, *, attention_backend: str = "torch"):
    Anima = load_anima_class()
    config = load_config_from_ckpt(ckpt_path)
    config["atten_backend"] = str(attention_backend or "torch")
    model = Anima(**config)
    state_dict = load_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        LOGGER.warning(
            "Anima model state mismatch: missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    model = model.to(device=device, dtype=dtype)
    return model


@dataclass
class AnimaVAEHandle:
    model: torch.nn.Module
    scale: list[torch.Tensor]

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def to(self, device=None, dtype=None):
        self.model = self.model.to(device=device, dtype=dtype)
        if device is not None:
            self.scale = [value.to(device=device) for value in self.scale]
        if dtype is not None:
            self.scale = [value.to(dtype=dtype) for value in self.scale]
        return self

    def eval(self):
        self.model.eval()
        return self

    def requires_grad_(self, requires_grad: bool):
        self.model.requires_grad_(requires_grad)
        return self


def _load_tensor_state(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return {key: handle.get_tensor(key) for key in handle.keys()}
    loaded = torch.load(str(path), map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"]
    return loaded


def load_vae(vae_path, device, dtype):
    WanVAE_ = load_wan_vae_class()
    config = dict(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    model = WanVAE_(**config)
    state_dict = _load_tensor_state(Path(vae_path))
    info = model.load_state_dict(state_dict, strict=False)
    missing = len(getattr(info, "missing_keys", []))
    unexpected = len(getattr(info, "unexpected_keys", []))
    if missing or unexpected:
        LOGGER.warning("VAE state mismatch: missing=%d unexpected=%d", missing, unexpected)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    mean = torch.tensor(
        [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        dtype=dtype,
        device=device,
    )
    std = torch.tensor(
        [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
        dtype=dtype,
        device=device,
    )
    scale = [mean, 1.0 / std]
    return AnimaVAEHandle(model=model, scale=scale)


def load_qwen_tokenizer(qwen_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)


def load_qwen_model(qwen_path: str, device, dtype):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        qwen_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_qwen(qwen_path, device, dtype):
    tokenizer = load_qwen_tokenizer(qwen_path)
    model = load_qwen_model(qwen_path, device, dtype)
    return tokenizer, model


def _validate_t5_tokenizer_dir(t5_dir: Path):
    if not t5_dir.exists():
        raise FileNotFoundError(f"T5 tokenizer directory not found: {t5_dir}")
    if not t5_dir.is_dir():
        raise NotADirectoryError(f"T5 tokenizer path is not a directory: {t5_dir}")

    missing = [name for name in T5_REQUIRED_FILES if not (t5_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"T5 tokenizer directory missing required files: {missing}")

    try:
        config = json.loads((t5_dir / "config.json").read_text(encoding="utf-8"))
        vocab_size = config.get("vocab_size")
        if vocab_size is not None and int(vocab_size) != int(T5_EXPECTED_VOCAB_SIZE):
            LOGGER.warning(
                "T5 tokenizer config vocab_size=%s (expected=%s)",
                vocab_size,
                T5_EXPECTED_VOCAB_SIZE,
            )
    except Exception as exc:
        LOGGER.warning("Failed to inspect T5 tokenizer config.json: %s", exc)


def load_t5_tokenizer(t5_dir: str):
    from transformers import T5TokenizerFast

    tokenizer_dir = Path(t5_dir).expanduser().resolve()
    _validate_t5_tokenizer_dir(tokenizer_dir)
    return T5TokenizerFast.from_pretrained(str(tokenizer_dir))


def tokenize_qwen(tokenizer, captions, max_length=512):
    return tokenizer(
        captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def tokenize_t5(tokenizer, captions, max_length=512):
    return tokenizer(
        captions,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
