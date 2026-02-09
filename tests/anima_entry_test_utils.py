from __future__ import annotations

import argparse
import importlib
import sys
import types


def _build_stub_train_network_module() -> types.ModuleType:
    module = types.ModuleType("train_network")

    class _NetworkTrainer:
        def __init__(self):
            pass

        def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
            del args
            if train_dataset_group is not None:
                train_dataset_group.verify_bucket_reso_steps(64)
            if val_dataset_group is not None:
                val_dataset_group.verify_bucket_reso_steps(64)

        def train(self, args):
            del args

        def build_resume_snapshot(self, args):
            del args
            return None

        def validate_resume_snapshot(self, args, snapshot):
            del args, snapshot
            return None

        def generate_step_logs(
            self,
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer=None,
            keys_scaled=None,
            mean_norm=None,
            maximum_norm=None,
            mean_grad_norm=None,
            mean_combined_norm=None,
        ):
            del (
                args,
                lr_scheduler,
                lr_descriptions,
                optimizer,
                keys_scaled,
                mean_norm,
                maximum_norm,
                mean_grad_norm,
                mean_combined_norm,
            )
            return {"loss/current": current_loss, "loss/average": avr_loss}

        def step_logging(self, accelerator, logs, global_step, epoch):
            del accelerator, logs, global_step, epoch
            return None

    def _setup_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument("--pretrained_model_name_or_path", type=str, default="")
        parser.add_argument("--vae", type=str, default="")
        parser.add_argument("--network_module", type=str, default=None)
        parser.add_argument("--network_args", type=str, nargs="*", default=None)
        parser.add_argument("--network_dim", type=int, default=None)
        parser.add_argument("--network_alpha", type=float, default=1.0)
        parser.add_argument("--network_dropout", type=float, default=None)
        parser.add_argument("--output_dir", type=str, default=None)
        parser.add_argument("--output_name", type=str, default=None)
        parser.add_argument("--save_every_n_steps", type=int, default=None)
        parser.add_argument("--save_state", action="store_true")
        parser.add_argument("--optimizer_type", type=str, default="AdamW")
        parser.add_argument("--optimizer_args", type=str, nargs="*", default=None)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--lr_scheduler", type=str, default="constant")
        parser.add_argument("--lr_warmup_steps", type=float, default=0)
        parser.add_argument("--lr_scheduler_num_cycles", type=float, default=1)
        parser.add_argument("--lr_scheduler_min_lr_ratio", type=float, default=None)
        parser.add_argument("--cache_text_encoder_outputs", action="store_true")
        parser.add_argument("--cache_text_encoder_outputs_to_disk", action="store_true")
        parser.add_argument("--cache_latents_to_disk", action="store_true")
        parser.add_argument("--vae_batch_size", type=int, default=1)
        parser.add_argument("--skip_cache_check", action="store_true")
        parser.add_argument("--bucket_reso_steps", type=int, default=64)
        parser.add_argument("--train_data_dir", type=str, default=None)
        parser.add_argument("--resolution", type=str, default=None)
        parser.add_argument("--min_bucket_reso", type=int, default=256)
        parser.add_argument("--max_bucket_reso", type=int, default=1024)
        parser.add_argument("--shuffle_caption", action="store_true")
        parser.add_argument("--keep_tokens", type=int, default=0)
        parser.add_argument("--flip_aug", action="store_true")
        parser.add_argument("--cache_latents", action="store_true")
        parser.add_argument("--max_train_epochs", type=int, default=None)
        parser.add_argument("--max_train_steps", type=int, default=1000)
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--mixed_precision", type=str, default="no")
        parser.add_argument("--gradient_checkpointing", action="store_true")
        parser.add_argument("--max_data_loader_n_workers", type=int, default=0)
        parser.add_argument("--xformers", action="store_true")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--caption_extension", type=str, default=".txt")
        return parser

    module.NetworkTrainer = _NetworkTrainer
    module.setup_parser = _setup_parser
    return module


def _install_stub_modules():
    fake_train_util = types.ModuleType("library.train_util")
    fake_train_util.verify_command_line_training_args = lambda args: args
    fake_train_util.read_config_from_file = lambda args, parser: args
    sys.modules["library.train_util"] = fake_train_util

    fake_utils = types.ModuleType("library.utils")
    fake_utils.setup_logging = lambda *args, **kwargs: None
    sys.modules["library.utils"] = fake_utils

    sys.modules["train_network"] = _build_stub_train_network_module()


def import_native_entry():
    _install_stub_modules()
    sys.modules.pop("anima_train_network", None)
    return importlib.import_module("anima_train_network")
