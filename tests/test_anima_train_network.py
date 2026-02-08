from __future__ import annotations

from pathlib import Path

import pytest

from library.anima_runtime.config_adapter import apply_root_anima_config_if_needed
from tests.anima_entry_test_utils import import_native_entry


class _DummyDatasetGroup:
    def verify_bucket_reso_steps(self, _value):
        return None


def _build_minimal_args():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args([])
    args.anima_transformer = "anima.safetensors"
    args.pretrained_model_name_or_path = "anima.safetensors"
    args.vae = "vae.safetensors"
    args.qwen = "qwen_path"
    args.t5_tokenizer_dir = "t5_dir"
    args.lr_scheduler = "constant"
    args.lr_warmup_steps = 0
    return parser, args


def test_bridge_only_args_are_removed():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--anima-train-py", "x.py"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--print-cmd"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--dry-run"])


def test_assert_extra_args_requires_t5():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.t5_tokenizer_dir = ""
    trainer = native_entry.AnimaNetworkTrainer()
    with pytest.raises(ValueError, match="T5 tokenizer directory is required"):
        trainer.assert_extra_args(args, _DummyDatasetGroup(), None)


def test_assert_extra_args_rejects_schedulefree_with_cosine():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.optimizer_type = "radam_schedulefree"
    args.lr_scheduler = "cosine"
    trainer = native_entry.AnimaNetworkTrainer()
    with pytest.raises(ValueError, match="requires lr_scheduler=constant"):
        trainer.assert_extra_args(args, _DummyDatasetGroup(), None)


def test_assert_extra_args_accepts_schedulefree_constant_no_warmup():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.optimizer_type = "radam_schedulefree"
    args.lr_scheduler = "constant"
    args.lr_warmup_steps = 0
    trainer = native_entry.AnimaNetworkTrainer()
    trainer.assert_extra_args(args, _DummyDatasetGroup(), None)
    assert args.optimizer_type == "RAdamScheduleFree"


def test_assert_extra_args_rejects_schedulefree_with_warmup_ratio():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.optimizer_type = "radam_schedulefree"
    args.lr_scheduler = "constant"
    args.lr_warmup_steps = 0
    args.lr_warmup_ratio = 0.1
    trainer = native_entry.AnimaNetworkTrainer()
    with pytest.raises(ValueError, match="requires lr_warmup_ratio=0"):
        trainer.assert_extra_args(args, _DummyDatasetGroup(), None)


def test_root_config_adapter_maps_network_module(tmp_path: Path):
    native_entry = import_native_entry()
    config_file = tmp_path / "anima_root.toml"
    config_file.write_text(
        """
[model]
transformer_path = "anima.safetensors"
vae_path = "vae.safetensors"
text_encoder_path = "qwen"
t5_tokenizer_path = "t5"

[dataset]
data_dir = "dataset"
resolution = 1024
min_reso = 512
max_reso = 1536
reso_step = 64

[training]
batch_size = 1

[lora]
network_type = "lokr"
lora_rank = 16
lora_alpha = 16
        """.strip(),
        encoding="utf-8",
    )

    parser = native_entry.setup_parser()
    args = parser.parse_args(["--config", str(config_file)])
    args = apply_root_anima_config_if_needed(args, parser)
    assert args.network_module == "networks.lokr_anima"
    assert args.network_dim == 16
    assert args.network_alpha == 16
