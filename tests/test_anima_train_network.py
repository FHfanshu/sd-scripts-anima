from __future__ import annotations

from pathlib import Path

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


def test_assert_extra_args_forces_gradient_checkpointing_enabled():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.gradient_checkpointing = False
    trainer = native_entry.AnimaNetworkTrainer()
    trainer.assert_extra_args(args, _DummyDatasetGroup(), None)
    assert args.gradient_checkpointing is True


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


def test_apply_anima_runtime_defaults_enable_tensorboard_with_experiment_prefix():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.output_dir = "E:/sd-scripts-anima/output/exp1"
    args.output_name = "newzx-anima-lokr"
    args.logging_dir = None
    args.log_with = None
    args.log_prefix = None

    native_entry.apply_anima_runtime_defaults(args)

    assert args.log_with == "tensorboard"
    assert str(args.logging_dir).replace("\\", "/").endswith("logs")
    assert args.log_prefix == "newzx-anima-lokr_"


def test_apply_anima_runtime_defaults_supports_legacy_xfomers_flag():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.xformers = False
    args.xfomers = True

    native_entry.apply_anima_runtime_defaults(args)

    assert args.xformers is True


def test_apply_anima_runtime_defaults_legacy_xfomers_false_does_not_disable_xformers():
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.xformers = True
    args.xfomers = False

    native_entry.apply_anima_runtime_defaults(args)

    assert args.xformers is True


def test_parser_auto_start_tensorboard_default_true():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args([])
    assert args.auto_start_tensorboard is True
    assert args.tensorboard_host == "127.0.0.1"
    assert args.tensorboard_port == 6006


def test_maybe_start_tensorboard_starts_when_enabled(monkeypatch, capsys):
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.log_with = "tensorboard"
    args.logging_dir = "output/test/logs"
    args.auto_start_tensorboard = True
    args.tensorboard_host = "127.0.0.1"
    args.tensorboard_port = 6123
    args.tensorboard_logdir = ""

    popen_calls = []

    class _DummyPopen:
        def __init__(self, cmd, **kwargs):
            popen_calls.append((cmd, kwargs))

    monkeypatch.setattr(native_entry, "_is_primary_process_for_side_effects", lambda: True)
    monkeypatch.setattr(native_entry, "_pick_available_port", lambda host, port, max_tries=20: 6123)
    monkeypatch.setattr(native_entry.subprocess, "Popen", _DummyPopen)

    url = native_entry.maybe_start_tensorboard(args)

    assert url == "http://127.0.0.1:6123"
    assert len(popen_calls) == 1
    assert "--logdir" in popen_calls[0][0]
    assert "output/test/logs" in popen_calls[0][0]
    out = capsys.readouterr().out
    assert "TensorBoard: http://127.0.0.1:6123" in out


def test_maybe_start_tensorboard_skips_when_log_with_not_tensorboard(monkeypatch):
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.log_with = "wandb"
    args.logging_dir = "output/test/logs"
    args.auto_start_tensorboard = True

    called = {"popen": False}

    def _fake_popen(*_args, **_kwargs):
        called["popen"] = True
        return None

    monkeypatch.setattr(native_entry.subprocess, "Popen", _fake_popen)

    url = native_entry.maybe_start_tensorboard(args)

    assert url is None
    assert called["popen"] is False


def test_maybe_start_tensorboard_skips_on_non_primary_process(monkeypatch):
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.log_with = "tensorboard"
    args.logging_dir = "output/test/logs"
    args.auto_start_tensorboard = True

    called = {"popen": False}

    def _fake_popen(*_args, **_kwargs):
        called["popen"] = True
        return None

    monkeypatch.setattr(native_entry, "_is_primary_process_for_side_effects", lambda: False)
    monkeypatch.setattr(native_entry.subprocess, "Popen", _fake_popen)

    url = native_entry.maybe_start_tensorboard(args)

    assert url is None
    assert called["popen"] is False


def test_maybe_start_tensorboard_falls_back_to_next_port(monkeypatch, caplog):
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.log_with = "tensorboard"
    args.logging_dir = "output/test/logs"
    args.auto_start_tensorboard = True
    args.tensorboard_host = "127.0.0.1"
    args.tensorboard_port = 6006
    args.tensorboard_logdir = ""

    class _DummyPopen:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(native_entry, "_is_primary_process_for_side_effects", lambda: True)
    monkeypatch.setattr(native_entry, "_pick_available_port", lambda host, port, max_tries=20: 6007)
    monkeypatch.setattr(native_entry.subprocess, "Popen", _DummyPopen)

    with caplog.at_level("WARNING"):
        url = native_entry.maybe_start_tensorboard(args)

    assert url == "http://127.0.0.1:6007"
    assert args.tensorboard_port == 6007
    assert "falling back to available port 6007" in caplog.text


def test_maybe_auto_convert_single_file_config_for_root_style(tmp_path: Path):
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    config_path = tmp_path / "single.toml"
    config_path.write_text(
        "\n".join(
            [
                "[model]",
                'transformer_path = "m.safetensors"',
                'vae_path = "v.safetensors"',
                'text_encoder_path = "qwen_dir"',
                't5_tokenizer_dir = "t5_dir"',
                "",
                "[dataset]",
                'data_dir = "/data/train/10_subject"',
                "repeats = 1",
                "resolution = 512",
                "",
                "[training]",
                "epochs = 1",
                "batch_size = 1",
                "grad_accum = 1",
                "learning_rate = 0.0001",
                'mixed_precision = "bf16"',
                "",
                "[lora]",
                'network_type = "lokr"',
                "lora_rank = 8",
                "lora_alpha = 8",
                "",
                "[optimizer]",
                'type = "AdamW"',
                "",
                "[output]",
                'output_dir = "./output"',
                'output_name = "test"',
            ]
        ),
        encoding="utf-8",
    )
    cli_argv = ["--config_file", str(config_path)]
    args = parser.parse_args(cli_argv)

    converted_args = native_entry.maybe_auto_convert_single_file_config(args, parser=parser, cli_argv=cli_argv)

    assert bool(getattr(converted_args, "_anima_single_file_converted", False))
    assert isinstance(converted_args.dataset_config, dict)
    assert "datasets" in converted_args.dataset_config
    assert str(converted_args.config_file).replace("\\", "/").endswith("/single")
    assert converted_args.network_module == "networks.lokr_anima"


def test_maybe_auto_convert_single_file_config_skips_non_root_style(tmp_path: Path):
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    config_path = tmp_path / "flat.toml"
    config_path.write_text('anima_transformer = "m.safetensors"\n', encoding="utf-8")
    cli_argv = ["--config_file", str(config_path)]
    args = parser.parse_args(cli_argv)

    converted_args = native_entry.maybe_auto_convert_single_file_config(args, parser=parser, cli_argv=cli_argv)

    assert converted_args.config_file == str(config_path)
    assert converted_args.dataset_config is None
    assert not bool(getattr(converted_args, "_anima_single_file_converted", False))


def test_maybe_auto_convert_single_file_config_skips_when_output_config_enabled(tmp_path: Path):
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    config_path = tmp_path / "single.toml"
    config_path.write_text(
        """
[model]
transformer_path = "m.safetensors"
vae_path = "v.safetensors"
text_encoder_path = "qwen_dir"
t5_tokenizer_dir = "t5_dir"

[training]
epochs = 1
        """.strip(),
        encoding="utf-8",
    )
    cli_argv = ["--config_file", str(config_path), "--output_config"]
    args = parser.parse_args(cli_argv)

    converted_args = native_entry.maybe_auto_convert_single_file_config(args, parser=parser, cli_argv=cli_argv)

    assert converted_args is args
    assert not bool(getattr(converted_args, "_anima_single_file_converted", False))


def test_maybe_auto_convert_single_file_config_prefers_cli_dataset_config(tmp_path: Path, caplog):
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    config_path = tmp_path / "single.toml"
    config_path.write_text(
        "\n".join(
            [
                "[model]",
                'transformer_path = "m.safetensors"',
                'vae_path = "v.safetensors"',
                'text_encoder_path = "qwen_dir"',
                't5_tokenizer_dir = "t5_dir"',
                "",
                "[dataset]",
                'data_dir = "/data/train/10_subject"',
                "repeats = 1",
                "",
                "[training]",
                "epochs = 1",
                "batch_size = 1",
                "grad_accum = 1",
                "learning_rate = 0.0001",
                "",
                "[lora]",
                'network_type = "lora"',
                "lora_rank = 8",
                "lora_alpha = 8",
                "",
                "[output]",
                'output_dir = "./output"',
                'output_name = "test"',
            ]
        ),
        encoding="utf-8",
    )
    cli_argv = ["--config_file", str(config_path), "--dataset_config", "custom_dataset.toml"]
    args = parser.parse_args(cli_argv)

    with caplog.at_level("WARNING"):
        converted_args = native_entry.maybe_auto_convert_single_file_config(args, parser=parser, cli_argv=cli_argv)

    assert bool(getattr(converted_args, "_anima_single_file_converted", False))
    assert converted_args.dataset_config == "custom_dataset.toml"
    assert "using --dataset_config and ignoring inline [dataset]" in caplog.text


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


def test_parser_t5_strict_validation_default_false():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args([])
    assert args.t5_tokenizer_validate_strict is False


def test_parser_t5_strict_validation_can_enable():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args(["--t5_tokenizer_validate_strict"])
    assert args.t5_tokenizer_validate_strict is True


def test_parser_anima_monitor_defaults():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args([])
    assert args.anima_monitor_memory is True
    assert args.anima_monitor_alert_policy == "warn"
    assert args.anima_monitor_memory_warn_ratio == pytest.approx(0.95)
    assert args.anima_monitor_loss_spike_ratio == pytest.approx(3.0)


def test_parser_anima_monitor_options():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args(
        [
            "--no-anima_monitor_memory",
            "--anima_monitor_alert_policy",
            "raise",
            "--anima_monitor_memory_warn_ratio",
            "0.9",
            "--anima_monitor_loss_spike_ratio",
            "2.5",
        ]
    )
    assert args.anima_monitor_memory is False
    assert args.anima_monitor_alert_policy == "raise"
    assert args.anima_monitor_memory_warn_ratio == pytest.approx(0.9)
    assert args.anima_monitor_loss_spike_ratio == pytest.approx(2.5)


def test_parser_anima_resume_snapshot_strict_default_false():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args([])
    assert args.anima_resume_snapshot_strict is False


def test_parser_anima_resume_snapshot_strict_can_enable():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args(["--anima_resume_snapshot_strict"])
    assert args.anima_resume_snapshot_strict is True


def test_parser_resume_ignore_optimizer_mismatch_default_true():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args([])
    assert args.resume_ignore_optimizer_mismatch is True


def test_parser_resume_ignore_optimizer_mismatch_can_disable():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args(["--no-resume_ignore_optimizer_mismatch"])
    assert args.resume_ignore_optimizer_mismatch is False


def test_assert_extra_args_xformers_flag_forces_anima_attention_backend(monkeypatch):
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.xformers = True
    args.anima_attention_backend = "torch"

    monkeypatch.setattr(native_entry, "_is_xformers_available", lambda: True)

    trainer = native_entry.AnimaNetworkTrainer()
    trainer.assert_extra_args(args, _DummyDatasetGroup(), None)
    assert args.anima_attention_backend == "xformers"
    assert args.xformers is True


def test_assert_extra_args_xformers_unavailable_falls_back_to_torch(monkeypatch):
    native_entry = import_native_entry()
    _parser, args = _build_minimal_args()
    args.xformers = True
    args.anima_attention_backend = "xformers"

    monkeypatch.setattr(native_entry, "_is_xformers_available", lambda: False)

    trainer = native_entry.AnimaNetworkTrainer()
    trainer.assert_extra_args(args, _DummyDatasetGroup(), None)
    assert args.anima_attention_backend == "torch"
    assert args.xformers is False
