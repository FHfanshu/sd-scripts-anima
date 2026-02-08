from __future__ import annotations

from pathlib import Path
from typing import Any

import toml


OPTIMIZER_ALIASES = {
    "radam_schedulefree": "RAdamScheduleFree",
    "adamw_schedulefree": "AdamWScheduleFree",
    "sgd_schedulefree": "SGDScheduleFree",
}


def _coerce_optimizer_type(value: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        return normalized
    lower = normalized.lower()
    if lower in OPTIMIZER_ALIASES:
        return OPTIMIZER_ALIASES[lower]
    return normalized


def normalize_optimizer_aliases(args):
    if getattr(args, "optimizer_type", None):
        args.optimizer_type = _coerce_optimizer_type(args.optimizer_type)
    return args


def _is_root_anima_config(config: dict[str, Any]) -> bool:
    return isinstance(config, dict) and "model" in config and "training" in config


def _should_override(current_value, default_value):
    if current_value is None:
        return True
    if isinstance(current_value, str):
        return current_value == "" or current_value == default_value
    return current_value == default_value


def _set_if_unset(args, defaults: dict[str, Any], key: str, value):
    if value is None or not hasattr(args, key):
        return
    if _should_override(getattr(args, key), defaults.get(key)):
        setattr(args, key, value)


def _append_network_arg(args, item: str):
    values = list(getattr(args, "network_args", None) or [])
    key = item.split("=", 1)[0].strip()
    values = [entry for entry in values if not str(entry).startswith(f"{key}=")]
    values.append(item)
    args.network_args = values


def apply_root_anima_config_if_needed(args, parser):
    config_path = str(getattr(args, "config", "") or "").strip()
    if not config_path:
        return args

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        parser.error(f"--config file does not exist: {config_file}")

    loaded = toml.load(str(config_file))
    if not _is_root_anima_config(loaded):
        return args

    defaults = vars(parser.parse_args([]))

    model = loaded.get("model", {}) or {}
    dataset = loaded.get("dataset", {}) or {}
    training = loaded.get("training", {}) or {}
    optimizer = loaded.get("optimizer", {}) or {}
    lora = loaded.get("lora", {}) or {}
    output = loaded.get("output", {}) or {}

    _set_if_unset(args, defaults, "anima_transformer", model.get("transformer_path"))
    _set_if_unset(args, defaults, "pretrained_model_name_or_path", model.get("transformer_path"))
    _set_if_unset(args, defaults, "vae", model.get("vae_path"))
    _set_if_unset(args, defaults, "qwen", model.get("text_encoder_path"))
    _set_if_unset(
        args,
        defaults,
        "t5_tokenizer_dir",
        model.get("t5_tokenizer_dir") or model.get("t5_tokenizer_path"),
    )

    _set_if_unset(args, defaults, "train_data_dir", dataset.get("data_dir"))
    if dataset.get("resolution") is not None:
        _set_if_unset(args, defaults, "resolution", str(dataset.get("resolution")))
    _set_if_unset(args, defaults, "min_bucket_reso", dataset.get("min_reso"))
    _set_if_unset(args, defaults, "max_bucket_reso", dataset.get("max_reso"))
    _set_if_unset(args, defaults, "bucket_reso_steps", dataset.get("reso_step"))
    _set_if_unset(args, defaults, "shuffle_caption", bool(dataset.get("shuffle_caption", False)))
    _set_if_unset(args, defaults, "keep_tokens", dataset.get("keep_tokens"))
    _set_if_unset(args, defaults, "flip_aug", bool(dataset.get("flip_augment", False)))
    _set_if_unset(args, defaults, "cache_latents", bool(dataset.get("cache_latents", False)))

    if training.get("epochs") is not None:
        _set_if_unset(args, defaults, "max_train_epochs", int(training.get("epochs")))
    if training.get("max_steps") is not None and int(training.get("max_steps")) > 0:
        _set_if_unset(args, defaults, "max_train_steps", int(training.get("max_steps")))
    if training.get("batch_size") is not None:
        _set_if_unset(args, defaults, "train_batch_size", int(training.get("batch_size")))
    if training.get("grad_accum") is not None:
        _set_if_unset(args, defaults, "gradient_accumulation_steps", int(training.get("grad_accum")))
    if training.get("learning_rate") is not None:
        _set_if_unset(args, defaults, "learning_rate", float(training.get("learning_rate")))
    if training.get("mixed_precision") is not None:
        _set_if_unset(args, defaults, "mixed_precision", str(training.get("mixed_precision")))
    if training.get("grad_checkpoint") is not None:
        _set_if_unset(args, defaults, "gradient_checkpointing", bool(training.get("grad_checkpoint")))
    if training.get("num_workers") is not None:
        _set_if_unset(args, defaults, "max_data_loader_n_workers", int(training.get("num_workers")))
    if training.get("xformers") is not None:
        _set_if_unset(args, defaults, "xformers", bool(training.get("xformers")))
    if training.get("seed") is not None:
        _set_if_unset(args, defaults, "seed", int(training.get("seed")))
    if training.get("seq_len") is not None:
        _set_if_unset(args, defaults, "anima_seq_len", int(training.get("seq_len")))
    if training.get("noise_offset") is not None:
        _set_if_unset(args, defaults, "anima_noise_offset", float(training.get("noise_offset")))

    scheduler = str(training.get("lr_scheduler", "") or "").strip()
    lr_min_ratio = training.get("lr_min_ratio")
    if scheduler:
        mapped_scheduler = scheduler
        if scheduler == "cosine" and lr_min_ratio is not None and float(lr_min_ratio) > 0:
            mapped_scheduler = "cosine_with_min_lr"
            _set_if_unset(args, defaults, "lr_scheduler_min_lr_ratio", float(lr_min_ratio))
        _set_if_unset(args, defaults, "lr_scheduler", mapped_scheduler)
    if training.get("lr_warmup_steps") is not None:
        _set_if_unset(args, defaults, "lr_warmup_steps", int(training.get("lr_warmup_steps")))
    if (
        training.get("lr_warmup_ratio") is not None
        and float(training.get("lr_warmup_ratio")) > 0
        and int(getattr(args, "lr_warmup_steps", 0) or 0) <= 0
    ):
        _set_if_unset(args, defaults, "lr_warmup_steps", float(training.get("lr_warmup_ratio")))
    if training.get("lr_scheduler_num_cycles") is not None:
        _set_if_unset(args, defaults, "lr_scheduler_num_cycles", int(training.get("lr_scheduler_num_cycles")))

    if optimizer.get("type") is not None:
        _set_if_unset(args, defaults, "optimizer_type", _coerce_optimizer_type(str(optimizer.get("type"))))
    optimizer_args = list(getattr(args, "optimizer_args", None) or [])
    if optimizer.get("weight_decay") is not None:
        optimizer_args.append(f"weight_decay={float(optimizer.get('weight_decay'))}")
    if optimizer.get("eps") is not None:
        optimizer_args.append(f"eps={float(optimizer.get('eps'))}")
    beta1 = optimizer.get("beta1")
    beta2 = optimizer.get("beta2")
    if beta1 is not None and beta2 is not None:
        optimizer_args.append(f"betas=({float(beta1)},{float(beta2)})")
    if optimizer_args and _should_override(getattr(args, "optimizer_args", None), defaults.get("optimizer_args")):
        args.optimizer_args = optimizer_args

    if lora.get("network_type") in {"lora", "lokr"} and _should_override(
        getattr(args, "network_module", None), defaults.get("network_module")
    ):
        args.network_module = "networks.lokr_anima" if str(lora.get("network_type")) == "lokr" else "networks.lora_anima"
    _set_if_unset(args, defaults, "network_dim", lora.get("lora_rank"))
    _set_if_unset(args, defaults, "network_alpha", lora.get("lora_alpha"))
    _set_if_unset(args, defaults, "network_dropout", lora.get("lora_dropout"))
    if lora.get("lora_targets"):
        _append_network_arg(args, f"target_modules={lora.get('lora_targets')}")
    if lora.get("lokr_factor") is not None:
        _append_network_arg(args, f"lokr_factor={int(lora.get('lokr_factor'))}")
    if lora.get("lokr_full_matrix") is not None:
        _append_network_arg(args, f"lokr_full_matrix={bool(lora.get('lokr_full_matrix'))}")
    if lora.get("lokr_decompose_both") is not None:
        _append_network_arg(args, f"lokr_decompose_both={bool(lora.get('lokr_decompose_both'))}")
    if lora.get("lokr_rank_dropout") is not None:
        _append_network_arg(args, f"lokr_rank_dropout={float(lora.get('lokr_rank_dropout'))}")
    if lora.get("lokr_module_dropout") is not None:
        _append_network_arg(args, f"lokr_module_dropout={float(lora.get('lokr_module_dropout'))}")

    _set_if_unset(args, defaults, "output_dir", output.get("output_dir"))
    _set_if_unset(args, defaults, "output_name", output.get("output_name") or lora.get("lora_name"))
    if output.get("save_every") is not None:
        _set_if_unset(args, defaults, "save_every_n_steps", int(output.get("save_every")))
    if output.get("save_state") is not None:
        _set_if_unset(args, defaults, "save_state", bool(output.get("save_state")))

    return normalize_optimizer_aliases(args)

