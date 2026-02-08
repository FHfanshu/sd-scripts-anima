from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


FORBIDDEN_TOKENS = (
    "ANIMA_TRAINER_ROOT",
    "ANIMA_TRAIN_PY",
    "../anima_train.py",
    "sys.path.insert(",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_no_parent_bridge_dependency_tokens():
    root = _repo_root()
    targets = [
        root / "anima_train_network.py",
        root / "library" / "anima_models.py",
        root / "library" / "anima_runtime" / "model_loading.py",
    ]
    for file_path in targets:
        text = file_path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TOKENS:
            assert token not in text, f"{file_path} should not contain: {token}"


def test_import_anima_train_network_without_parent_path():
    root = _repo_root()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    script = """
import argparse
import sys
import types

fake_train_util = types.ModuleType("library.train_util")
fake_train_util.verify_command_line_training_args = lambda args: args
fake_train_util.read_config_from_file = lambda args, parser: args
sys.modules["library.train_util"] = fake_train_util

fake_utils = types.ModuleType("library.utils")
fake_utils.setup_logging = lambda *args, **kwargs: None
sys.modules["library.utils"] = fake_utils

module = types.ModuleType("train_network")

class _NetworkTrainer:
    def __init__(self):
        pass
    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        pass

def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", default="")
    parser.add_argument("--network_module", default=None)
    parser.add_argument("--optimizer_type", default="AdamW")
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=float, default=0)
    parser.add_argument("--cache_text_encoder_outputs", action="store_true")
    parser.add_argument("--cache_text_encoder_outputs_to_disk", action="store_true")
    parser.add_argument("--bucket_reso_steps", type=int, default=64)
    parser.add_argument("--cache_latents_to_disk", action="store_true")
    parser.add_argument("--vae_batch_size", type=int, default=1)
    parser.add_argument("--skip_cache_check", action="store_true")
    parser.add_argument("--caption_extension", default=".txt")
    return parser

module.NetworkTrainer = _NetworkTrainer
module.setup_parser = _setup_parser
sys.modules["train_network"] = module

import anima_train_network
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
