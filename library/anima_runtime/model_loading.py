from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import torch

LOGGER = logging.getLogger(__name__)

T5_EXPECTED_VOCAB_SIZE = 32128
T5_REQUIRED_FILES = ("config.json", "spiece.model", "tokenizer.json")
T5_OPTIONAL_FILES = ("tokenizer_config.json", "special_tokens_map.json")
T5_DEFAULT_REPO_ID = "google/t5-v1_1-base"
T5_DEFAULT_SUBFOLDER = "tokenizer"
T5_DEFAULT_REVISION = "main"
T5_MODELSCOPE_FALLBACK_REPO_ID = "nv-community/Cosmos-Predict2-2B-Text2Image"
T5_MODELSCOPE_FALLBACK_REVISION = "master"
T5_MODELSCOPE_FALLBACK_SUBFOLDER = "tokenizer"


@dataclass
class _RepoSource:
    provider: str
    repo_id: str
    revision: str
    subfolder: str


def _normalize_subfolder(subfolder: Optional[str]) -> str:
    return str(subfolder or "").strip().strip("/")


def _parse_t5_repo_source(spec: str, *, default_provider: str, default_revision: str, default_subfolder: str) -> _RepoSource:
    text = str(spec or "").strip()
    if not text:
        raise ValueError("T5 repo source cannot be empty.")

    default_subfolder = _normalize_subfolder(default_subfolder)
    if "://" not in text:
        return _RepoSource(
            provider=default_provider,
            repo_id=text,
            revision=str(default_revision or T5_DEFAULT_REVISION),
            subfolder=default_subfolder,
        )

    parsed = urlparse(text)
    host = parsed.netloc.lower()
    parts = [p for p in parsed.path.split("/") if p]

    if "huggingface.co" in host:
        if len(parts) < 2:
            raise ValueError(f"Invalid HuggingFace URL for T5 source: {text}")
        repo_id = f"{parts[0]}/{parts[1]}"
        revision = str(default_revision or T5_DEFAULT_REVISION)
        subfolder = default_subfolder
        if len(parts) >= 4 and parts[2] in ("tree", "blob", "resolve"):
            revision = parts[3] or revision
            inferred_subfolder = _normalize_subfolder("/".join(parts[4:]))
            if inferred_subfolder:
                subfolder = inferred_subfolder
        return _RepoSource(provider="hf", repo_id=repo_id, revision=revision, subfolder=subfolder)

    if "modelscope.cn" in host:
        repo_id = ""
        revision = str(default_revision or T5_MODELSCOPE_FALLBACK_REVISION)
        subfolder = default_subfolder

        if "models" in parts:
            idx = parts.index("models")
            if len(parts) >= idx + 3:
                repo_id = f"{parts[idx + 1]}/{parts[idx + 2]}"
                tail = parts[idx + 3 :]
                if tail and tail[0] in ("tree", "resolve"):
                    if len(tail) >= 2 and tail[1]:
                        revision = tail[1]
                    inferred_subfolder = _normalize_subfolder("/".join(tail[2:]))
                    if inferred_subfolder:
                        subfolder = inferred_subfolder

        if not repo_id:
            raise ValueError(f"Invalid ModelScope URL for T5 source: {text}")

        q = parse_qs(parsed.query)
        if q.get("Revision"):
            revision = str(q["Revision"][0] or revision)
        if q.get("FilePath"):
            file_path = str(q["FilePath"][0] or "")
            inferred = _normalize_subfolder(str(Path(file_path).parent).replace("\\", "/"))
            if inferred and inferred != ".":
                subfolder = inferred

        return _RepoSource(provider="modelscope", repo_id=repo_id, revision=revision, subfolder=subfolder)

    raise ValueError(f"Unsupported URL host for T5 source: {text}")


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


def _missing_t5_tokenizer_files(t5_dir: Path):
    if not t5_dir.exists() or not t5_dir.is_dir():
        return list(T5_REQUIRED_FILES)
    return [name for name in T5_REQUIRED_FILES if not (t5_dir / name).exists()]


def _download_file(url: str, dst_path: Path):
    import requests

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dst_path.write_bytes(response.content)


def _download_t5_tokenizer_from_hf(t5_dir: Path, *, repo_id: str, revision: str, subfolder: str):
    from transformers import AutoConfig, T5TokenizerFast

    t5_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    normalized_subfolder = _normalize_subfolder(subfolder)
    if normalized_subfolder:
        candidates.append(normalized_subfolder)
    candidates.append("")

    errors = []
    for candidate in candidates:
        kwargs = {"revision": revision}
        if candidate:
            kwargs["subfolder"] = candidate
        try:
            tokenizer = T5TokenizerFast.from_pretrained(repo_id, **kwargs)
            tokenizer.save_pretrained(str(t5_dir))

            config = None
            for config_subfolder in ([candidate, ""] if candidate else [""]):
                config_kwargs = {"revision": revision}
                if config_subfolder:
                    config_kwargs["subfolder"] = config_subfolder
                try:
                    config = AutoConfig.from_pretrained(repo_id, **config_kwargs)
                    break
                except Exception:
                    continue

            if config is not None:
                config.save_pretrained(str(t5_dir))
            else:
                fallback_config = {"model_type": "t5", "vocab_size": int(getattr(tokenizer, "vocab_size", T5_EXPECTED_VOCAB_SIZE))}
                (t5_dir / "config.json").write_text(json.dumps(fallback_config), encoding="utf-8")
            return
        except Exception as exc:
            errors.append(f"subfolder={candidate or '<root>'}: {exc}")

    raise RuntimeError(
        f"Failed to download T5 tokenizer from Hugging Face repo={repo_id} revision={revision}. "
        + " | ".join(errors)
    )


def _download_t5_tokenizer_from_modelscope(t5_dir: Path, *, repo_id: str, revision: str, subfolder: str):
    t5_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://www.modelscope.cn/models/{repo_id}/resolve/{revision}"
    normalized_subfolder = _normalize_subfolder(subfolder)
    if normalized_subfolder:
        base = f"{base}/{normalized_subfolder}"

    required_errors = []
    for filename in T5_REQUIRED_FILES:
        try:
            _download_file(f"{base}/{filename}", t5_dir / filename)
        except Exception as exc:
            required_errors.append(f"{filename}: {exc}")

    if required_errors:
        raise RuntimeError(
            f"Failed to download required T5 tokenizer files from ModelScope repo={repo_id} "
            f"revision={revision} subfolder={normalized_subfolder or '<root>'}. "
            + " | ".join(required_errors)
        )

    for filename in T5_OPTIONAL_FILES:
        try:
            _download_file(f"{base}/{filename}", t5_dir / filename)
        except Exception as exc:
            LOGGER.warning("Optional T5 tokenizer file not downloaded from ModelScope (%s): %s", filename, exc)


def _download_t5_tokenizer_assets(
    t5_dir: Path,
    repo_id: str,
    *,
    repo_subfolder: str = T5_DEFAULT_SUBFOLDER,
    modelscope_fallback: bool = True,
    modelscope_repo_id: str = T5_MODELSCOPE_FALLBACK_REPO_ID,
    modelscope_revision: str = T5_MODELSCOPE_FALLBACK_REVISION,
    modelscope_subfolder: str = T5_MODELSCOPE_FALLBACK_SUBFOLDER,
):
    source = _parse_t5_repo_source(
        repo_id,
        default_provider="hf",
        default_revision=T5_DEFAULT_REVISION,
        default_subfolder=repo_subfolder,
    )

    if source.provider == "modelscope":
        _download_t5_tokenizer_from_modelscope(
            t5_dir,
            repo_id=source.repo_id,
            revision=source.revision,
            subfolder=source.subfolder,
        )
        return

    try:
        _download_t5_tokenizer_from_hf(
            t5_dir,
            repo_id=source.repo_id,
            revision=source.revision,
            subfolder=source.subfolder,
        )
    except Exception as hf_exc:
        if not modelscope_fallback:
            raise RuntimeError(f"Hugging Face T5 tokenizer download failed: {hf_exc}") from hf_exc

        ms_source = _parse_t5_repo_source(
            modelscope_repo_id,
            default_provider="modelscope",
            default_revision=modelscope_revision,
            default_subfolder=modelscope_subfolder,
        )
        try:
            _download_t5_tokenizer_from_modelscope(
                t5_dir,
                repo_id=ms_source.repo_id,
                revision=ms_source.revision,
                subfolder=ms_source.subfolder,
            )
            LOGGER.warning(
                "Falling back to ModelScope for T5 tokenizer download: repo=%s revision=%s subfolder=%s",
                ms_source.repo_id,
                ms_source.revision,
                ms_source.subfolder or "<root>",
            )
        except Exception as ms_exc:
            raise RuntimeError(
                "T5 tokenizer download failed from both Hugging Face and ModelScope. "
                f"HF error: {hf_exc} ; ModelScope error: {ms_exc}"
            ) from ms_exc

    missing = _missing_t5_tokenizer_files(t5_dir)
    if missing:
        raise FileNotFoundError(
            f"Auto-downloaded T5 tokenizer still missing required files: {missing} (dir={t5_dir}, repo={repo_id})"
        )


def load_t5_tokenizer(
    t5_dir: str,
    *,
    auto_download: bool = True,
    repo_id: str = T5_DEFAULT_REPO_ID,
    repo_subfolder: str = T5_DEFAULT_SUBFOLDER,
    modelscope_fallback: bool = True,
    modelscope_repo_id: str = T5_MODELSCOPE_FALLBACK_REPO_ID,
    modelscope_revision: str = T5_MODELSCOPE_FALLBACK_REVISION,
    modelscope_subfolder: str = T5_MODELSCOPE_FALLBACK_SUBFOLDER,
):
    from transformers import T5TokenizerFast

    tokenizer_dir = Path(t5_dir).expanduser().resolve()
    missing = _missing_t5_tokenizer_files(tokenizer_dir)
    if missing:
        if not auto_download:
            _validate_t5_tokenizer_dir(tokenizer_dir)
        LOGGER.warning(
            "T5 tokenizer files missing in %s: %s. Auto-downloading from source=%s (subfolder=%s)",
            tokenizer_dir,
            missing,
            repo_id,
            _normalize_subfolder(repo_subfolder) or "<root>",
        )
        _download_t5_tokenizer_assets(
            tokenizer_dir,
            repo_id=repo_id,
            repo_subfolder=repo_subfolder,
            modelscope_fallback=modelscope_fallback,
            modelscope_repo_id=modelscope_repo_id,
            modelscope_revision=modelscope_revision,
            modelscope_subfolder=modelscope_subfolder,
        )

    _validate_t5_tokenizer_dir(tokenizer_dir)
    return T5TokenizerFast.from_pretrained(str(tokenizer_dir), local_files_only=True)


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
