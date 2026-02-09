from __future__ import annotations

import pytest

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
    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "legacy_root_style.toml"])


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


def test_assert_extra_args_is_robust_when_optional_cache_flags_missing():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    if hasattr(args, "cache_text_encoder_outputs"):
        delattr(args, "cache_text_encoder_outputs")
    if hasattr(args, "cache_text_encoder_outputs_to_disk"):
        delattr(args, "cache_text_encoder_outputs_to_disk")
    trainer = native_entry.AnimaNetworkTrainer()
    trainer.assert_extra_args(args, _DummyDatasetGroup(), None)


def test_assert_extra_args_disables_gradient_checkpointing():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.gradient_checkpointing = True
    trainer = native_entry.AnimaNetworkTrainer()
    trainer.assert_extra_args(args, _DummyDatasetGroup(), None)
    assert args.gradient_checkpointing is False


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


def test_parser_train_norm_default_true():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args([])
    assert args.train_norm is True


def test_parser_allows_disabling_train_norm():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args(["--no-train_norm"])
    assert args.train_norm is False


def test_apply_anima_network_defaults_injects_train_norm_arg():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.train_norm = False
    args.network_args = ["foo=bar", "train_norm=true"]
    native_entry.apply_anima_network_defaults(args)
    assert args.network_args is not None
    assert "foo=bar" in args.network_args
    assert "train_norm=false" in args.network_args
    assert not any(arg == "train_norm=true" for arg in args.network_args)


def test_parser_t5_modelscope_fallback_default_true():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args([])
    assert args.t5_tokenizer_modelscope_fallback is True


def test_parser_t5_modelscope_fallback_can_disable():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args(["--no-t5_tokenizer_modelscope_fallback"])
    assert args.t5_tokenizer_modelscope_fallback is False
