from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import toml


def _load_converter_module():
    root = Path(__file__).resolve().parents[1]
    converter_path = root / "tools" / "convert_anima_root_to_kohya.py"
    spec = importlib.util.spec_from_file_location("convert_anima_root_to_kohya", converter_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_root_file_outputs_kohya_configs(tmp_path: Path):
    module = _load_converter_module()
    input_cfg = tmp_path / "root_anima.toml"
    input_cfg.write_text(
        """
[model]
transformer_path = "anima.safetensors"
vae_path = "vae.safetensors"
text_encoder_path = "qwen_dir"
t5_tokenizer_path = "t5_dir"

[dataset]
data_dir = "train_data"
resolution = 1024
min_reso = 512
max_reso = 1536
reso_step = 64
max_ar = 4
repeats = 2
shuffle_caption = true
keep_tokens = 1
caption_extension = ".txt"
flip_augment = false
cache_latents = true

[lora]
network_type = "lokr"
lora_rank = 32
lora_alpha = 32
lokr_factor = 8
lokr_full_matrix = true

[training]
epochs = 16
max_steps = 10
batch_size = 2
grad_accum = 3
learning_rate = 0.0001
lr_scheduler = "cosine"
lr_min_ratio = 0.1
lr_warmup_ratio = 0.03
lr_scheduler_num_cycles = 0.5
mixed_precision = "bf16"
num_workers = 2
seed = 42
seq_len = 256
noise_offset = 0.01

[optimizer]
type = "radam_schedulefree"
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
eps = 1e-08

[output]
output_dir = "./out"
output_name = "anima_test"
save_every = 5
save_state = true
        """.strip(),
        encoding="utf-8",
    )

    out_dir = tmp_path / "converted"
    train_args_path, dataset_path, unmapped = module.convert_root_file(
        input_path=input_cfg,
        output_dir=out_dir,
        train_args_name="train_args.toml",
        dataset_name="dataset.toml",
        overwrite=False,
    )

    train_args = toml.load(str(train_args_path))
    dataset_cfg = toml.load(str(dataset_path))

    assert train_args["anima_transformer"] == "anima.safetensors"
    assert train_args["network_module"] == "networks.lokr_anima"
    assert train_args["max_train_steps"] == 10
    assert "max_train_epochs" not in train_args
    assert train_args["lr_scheduler"] == "cosine_with_min_lr"
    assert train_args["lr_scheduler_num_cycles"] == 0.5
    assert train_args["optimizer_type"] == "RAdamScheduleFree"
    assert train_args["dataset_config"] == "dataset.toml"

    assert dataset_cfg["datasets"][0]["enable_bucket"] is True
    assert dataset_cfg["datasets"][0]["subsets"][0]["image_dir"] == "train_data"
    assert dataset_cfg["datasets"][0]["subsets"][0]["num_repeats"] == 2
    assert "dataset.max_ar" in unmapped


def test_convert_root_file_requires_t5_path(tmp_path: Path):
    module = _load_converter_module()
    input_cfg = tmp_path / "root_missing_t5.toml"
    input_cfg.write_text(
        """
[model]
transformer_path = "anima.safetensors"
vae_path = "vae.safetensors"
text_encoder_path = "qwen_dir"

[training]
max_steps = 10
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing model.t5_tokenizer_dir"):
        module.convert_root_file(
            input_path=input_cfg,
            output_dir=tmp_path / "converted",
            train_args_name="train_args.toml",
            dataset_name="dataset.toml",
            overwrite=False,
        )
