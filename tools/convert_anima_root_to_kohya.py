from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import toml

from library.anima_runtime.config_adapter import coerce_optimizer_type


def _strip_text(value: Any) -> str:
    return str(value or "").strip()


def _is_non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _set_if_present(dst: Dict[str, Any], key: str, value: Any):
    if _is_non_empty(value):
        dst[key] = value


def _map_network_module(network_type: str) -> str:
    if _strip_text(network_type).lower() == "lokr":
        return "networks.lokr_anima"
    return "networks.lora_anima"


def convert_root_to_kohya_dicts(root_cfg: Dict[str, Any], dataset_filename: str = "dataset.toml") -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    if not isinstance(root_cfg, dict) or "model" not in root_cfg or "training" not in root_cfg:
        raise ValueError("Input config is not a valid root-style Anima TOML. Expected sections: [model], [training].")

    model = root_cfg.get("model", {}) or {}
    dataset = root_cfg.get("dataset", {}) or {}
    training = root_cfg.get("training", {}) or {}
    optimizer = root_cfg.get("optimizer", {}) or {}
    lora = root_cfg.get("lora", {}) or {}
    output = root_cfg.get("output", {}) or {}

    train_args: Dict[str, Any] = {}
    dataset_cfg: Dict[str, Any] = {}
    unmapped_keys: List[str] = []

    transformer_path = model.get("transformer_path")
    _set_if_present(train_args, "anima_transformer", transformer_path)
    _set_if_present(train_args, "pretrained_model_name_or_path", transformer_path)
    _set_if_present(train_args, "vae", model.get("vae_path"))
    _set_if_present(train_args, "qwen", model.get("text_encoder_path"))
    t5_tokenizer_dir = model.get("t5_tokenizer_dir") or model.get("t5_tokenizer_path")
    if not _is_non_empty(t5_tokenizer_dir):
        raise ValueError(
            "Input config is missing model.t5_tokenizer_dir/model.t5_tokenizer_path. "
            "LLMAdapter training requires T5 tokenizer."
        )
    train_args["t5_tokenizer_dir"] = str(t5_tokenizer_dir)

    network_type = _strip_text(lora.get("network_type") or "lora")
    train_args["network_module"] = _map_network_module(network_type)
    _set_if_present(train_args, "network_dim", lora.get("lora_rank"))
    _set_if_present(train_args, "network_alpha", lora.get("lora_alpha"))
    _set_if_present(train_args, "network_dropout", lora.get("lora_dropout"))

    network_args: List[str] = []
    if _is_non_empty(lora.get("lora_targets")):
        network_args.append(f"target_modules={lora.get('lora_targets')}")
    if lora.get("lokr_factor") is not None:
        network_args.append(f"lokr_factor={int(lora.get('lokr_factor'))}")
    if lora.get("lokr_full_matrix") is not None:
        network_args.append(f"lokr_full_matrix={bool(lora.get('lokr_full_matrix'))}")
    if lora.get("lokr_decompose_both") is not None:
        network_args.append(f"lokr_decompose_both={bool(lora.get('lokr_decompose_both'))}")
    if lora.get("lokr_rank_dropout") is not None:
        network_args.append(f"lokr_rank_dropout={float(lora.get('lokr_rank_dropout'))}")
    if lora.get("lokr_module_dropout") is not None:
        network_args.append(f"lokr_module_dropout={float(lora.get('lokr_module_dropout'))}")
    if network_args:
        train_args["network_args"] = network_args

    if output:
        _set_if_present(train_args, "output_dir", output.get("output_dir"))
        _set_if_present(train_args, "output_name", output.get("output_name") or lora.get("lora_name"))
        if output.get("save_every") is not None:
            train_args["save_every_n_steps"] = int(output.get("save_every"))
        if output.get("save_state") is not None:
            train_args["save_state"] = bool(output.get("save_state"))
        if output.get("save_state_every") is not None:
            unmapped_keys.append("output.save_state_every")

    if training:
        max_steps = int(training.get("max_steps") or 0)
        if max_steps > 0:
            train_args["max_train_steps"] = max_steps
        elif training.get("epochs") is not None:
            train_args["max_train_epochs"] = int(training.get("epochs"))

        _set_if_present(train_args, "gradient_accumulation_steps", training.get("grad_accum"))
        _set_if_present(train_args, "learning_rate", training.get("learning_rate"))
        _set_if_present(train_args, "mixed_precision", training.get("mixed_precision"))
        if training.get("grad_checkpoint") is not None:
            train_args["gradient_checkpointing"] = bool(training.get("grad_checkpoint"))
        _set_if_present(train_args, "max_data_loader_n_workers", training.get("num_workers"))
        if training.get("persistent_data_loader_workers") is not None:
            train_args["persistent_data_loader_workers"] = bool(training.get("persistent_data_loader_workers"))
        if training.get("xformers") is not None:
            train_args["xformers"] = bool(training.get("xformers"))
        _set_if_present(train_args, "seed", training.get("seed"))
        _set_if_present(train_args, "anima_seq_len", training.get("seq_len"))
        _set_if_present(train_args, "anima_noise_offset", training.get("noise_offset"))
        _set_if_present(train_args, "caption_tag_dropout_rate", training.get("caption_tag_dropout_rate"))
        if _is_non_empty(training.get("resume")):
            train_args["resume"] = training.get("resume")

        scheduler = _strip_text(training.get("lr_scheduler"))
        if scheduler:
            if scheduler == "cosine" and float(training.get("lr_min_ratio") or 0) > 0:
                train_args["lr_scheduler"] = "cosine_with_min_lr"
                train_args["lr_scheduler_min_lr_ratio"] = float(training.get("lr_min_ratio"))
            else:
                train_args["lr_scheduler"] = scheduler

        if training.get("lr_warmup_steps") is not None:
            train_args["lr_warmup_steps"] = int(training.get("lr_warmup_steps"))
        elif float(training.get("lr_warmup_ratio") or 0) > 0:
            train_args["lr_warmup_steps"] = float(training.get("lr_warmup_ratio"))

        if training.get("lr_scheduler_num_cycles") is not None:
            train_args["lr_scheduler_num_cycles"] = float(training.get("lr_scheduler_num_cycles"))

        for key in ("auto_install", "train_parts", "text_cache_size", "log_every", "max_token_length"):
            if key in training:
                unmapped_keys.append(f"training.{key}")

    if optimizer:
        if _is_non_empty(optimizer.get("type")):
            train_args["optimizer_type"] = coerce_optimizer_type(str(optimizer.get("type")))
        optimizer_args: List[str] = []
        if optimizer.get("weight_decay") is not None:
            optimizer_args.append(f"weight_decay={float(optimizer.get('weight_decay'))}")
        if optimizer.get("eps") is not None:
            optimizer_args.append(f"eps={float(optimizer.get('eps'))}")
        beta1 = optimizer.get("beta1")
        beta2 = optimizer.get("beta2")
        if beta1 is not None and beta2 is not None:
            optimizer_args.append(f"betas=({float(beta1)},{float(beta2)})")
        if optimizer_args:
            train_args["optimizer_args"] = optimizer_args

    general: Dict[str, Any] = {}
    if dataset.get("shuffle_caption") is not None:
        general["shuffle_caption"] = bool(dataset.get("shuffle_caption"))
    if dataset.get("keep_tokens") is not None:
        general["keep_tokens"] = int(dataset.get("keep_tokens"))
    if _is_non_empty(dataset.get("keep_tokens_separator")):
        general["keep_tokens_separator"] = str(dataset.get("keep_tokens_separator"))
    if _is_non_empty(dataset.get("caption_extension")):
        general["caption_extension"] = str(dataset.get("caption_extension"))
    if dataset.get("flip_augment") is not None:
        general["flip_aug"] = bool(dataset.get("flip_augment"))
    if training.get("caption_tag_dropout_rate") is not None:
        general["caption_tag_dropout_rate"] = float(training.get("caption_tag_dropout_rate"))
    if general:
        dataset_cfg["general"] = general

    datasets_list: List[Dict[str, Any]] = []
    ds_item: Dict[str, Any] = {}
    if dataset.get("resolution") is not None:
        ds_item["resolution"] = int(dataset.get("resolution"))
    if training.get("batch_size") is not None:
        ds_item["batch_size"] = int(training.get("batch_size"))

    has_bucket_cfg = any(k in dataset for k in ("min_reso", "max_reso", "reso_step", "max_ar"))
    if has_bucket_cfg:
        ds_item["enable_bucket"] = True
        if dataset.get("min_reso") is not None:
            ds_item["min_bucket_reso"] = int(dataset.get("min_reso"))
        if dataset.get("max_reso") is not None:
            ds_item["max_bucket_reso"] = int(dataset.get("max_reso"))
        if dataset.get("reso_step") is not None:
            ds_item["bucket_reso_steps"] = int(dataset.get("reso_step"))

    subset: Dict[str, Any] = {}
    _set_if_present(subset, "image_dir", dataset.get("data_dir"))
    subset["num_repeats"] = int(dataset.get("repeats") or 1)

    if _is_non_empty(dataset.get("cache_dir")):
        unmapped_keys.append("dataset.cache_dir")
    if dataset.get("cache_latents") is not None:
        train_args["cache_latents"] = bool(dataset.get("cache_latents"))
    if dataset.get("max_ar") is not None:
        unmapped_keys.append("dataset.max_ar")

    ds_item["subsets"] = [subset]
    datasets_list.append(ds_item)
    dataset_cfg["datasets"] = datasets_list

    train_args["dataset_config"] = dataset_filename

    # root sections we do not map to kohya runtime options
    for sec_key in ("monitor", "wandb"):
        if sec_key in root_cfg:
            unmapped_keys.append(sec_key)

    return train_args, dataset_cfg, sorted(set(unmapped_keys))


def convert_root_file(
    input_path: Path,
    output_dir: Path,
    train_args_name: str,
    dataset_name: str,
    overwrite: bool = False,
) -> Tuple[Path, Path, List[str]]:
    root_cfg = toml.load(str(input_path))
    train_args, dataset_cfg, unmapped = convert_root_to_kohya_dicts(root_cfg, dataset_filename=dataset_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_args_path = output_dir / train_args_name
    dataset_path = output_dir / dataset_name

    if not overwrite:
        for p in (train_args_path, dataset_path):
            if p.exists():
                raise FileExistsError(f"Output file already exists: {p}")

    with train_args_path.open("w", encoding="utf-8") as f:
        toml.dump(train_args, f)
    with dataset_path.open("w", encoding="utf-8") as f:
        toml.dump(dataset_cfg, f)

    return train_args_path, dataset_path, unmapped


def main():
    parser = argparse.ArgumentParser(description="Convert root-style Anima TOML to Kohya-compatible train args and dataset config.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the root-style Anima TOML file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory for generated files. Default: <input_dir>/kohya_converted",
    )
    parser.add_argument("--train_args_name", type=str, default="train_args.toml", help="Output filename for train args TOML.")
    parser.add_argument("--dataset_name", type=str, default="dataset.toml", help="Output filename for dataset TOML.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input config does not exist: {input_path}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else input_path.parent / "kohya_converted"
    train_args_path, dataset_path, unmapped = convert_root_file(
        input_path=input_path,
        output_dir=output_dir,
        train_args_name=args.train_args_name,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
    )

    print(f"Generated: {train_args_path}")
    print(f"Generated: {dataset_path}")
    if unmapped:
        print("Unmapped keys:")
        for key in unmapped:
            print(f"  - {key}")


if __name__ == "__main__":
    main()
