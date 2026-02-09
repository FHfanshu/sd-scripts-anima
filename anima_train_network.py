from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler

import train_network
from library import strategy_base, train_util
from library.anima_models import (
    load_anima_qwen_model,
    load_anima_qwen_tokenizer,
    load_anima_t5_tokenizer,
    load_anima_transformer,
    load_anima_vae,
)
from library.anima_runtime.config_adapter import normalize_optimizer_aliases
from library.anima_runtime.flow_training import build_noisy_latents, compute_qwen_embeddings, sample_t
from library.strategy_anima import AnimaLatentsCachingStrategy, AnimaTextEncodingStrategy, AnimaTokenizeStrategy
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

SUPPORTED_NETWORK_MODULES = ("networks.lora_anima", "networks.lokr_anima")


def _upsert_network_arg(args, key: str, value: str):
    current = list(getattr(args, "network_args", None) or [])
    prefix = f"{key}="
    current = [arg for arg in current if not str(arg).startswith(prefix)]
    current.append(f"{key}={value}")
    args.network_args = current


def apply_anima_network_defaults(args):
    train_norm = bool(getattr(args, "train_norm", True))
    _upsert_network_arg(args, "train_norm", "true" if train_norm else "false")
    return args


class AnimaNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.qwen_tokenizer = None
        self.t5_tokenizer = None
        self._sample_warned = False
        self._runtime_args: Optional[argparse.Namespace] = None
        self._previous_loss_for_monitor: Optional[float] = None
        self._memory_metric_warning_logged = False

    @staticmethod
    def _to_optional_int(value):
        if value is None or str(value).strip() == "":
            return None
        return int(value)

    @staticmethod
    def _to_optional_float(value):
        if value is None or str(value).strip() == "":
            return None
        return float(value)

    def _build_resume_snapshot_fields(self, args: argparse.Namespace) -> Dict[str, Any]:
        return {
            "network_module": str(getattr(args, "network_module", "") or ""),
            "network_dim": self._to_optional_int(getattr(args, "network_dim", None)),
            "network_alpha": self._to_optional_float(getattr(args, "network_alpha", None)),
            "train_norm": bool(getattr(args, "train_norm", True)),
            "optimizer_type": str(getattr(args, "optimizer_type", "") or ""),
            "lr_scheduler": str(getattr(args, "lr_scheduler", "") or ""),
            "train_batch_size": self._to_optional_int(getattr(args, "train_batch_size", None)),
            "gradient_accumulation_steps": self._to_optional_int(getattr(args, "gradient_accumulation_steps", None)),
            "mixed_precision": str(getattr(args, "mixed_precision", "") or ""),
            "anima_seq_len": self._to_optional_int(getattr(args, "anima_seq_len", None)),
            "t5_tokenizer_repo_id": str(getattr(args, "t5_tokenizer_repo_id", "") or ""),
            "t5_tokenizer_subfolder": str(getattr(args, "t5_tokenizer_subfolder", "") or ""),
            "t5_tokenizer_modelscope_fallback": bool(getattr(args, "t5_tokenizer_modelscope_fallback", True)),
            "t5_tokenizer_modelscope_repo_id": str(getattr(args, "t5_tokenizer_modelscope_repo_id", "") or ""),
            "t5_tokenizer_modelscope_revision": str(getattr(args, "t5_tokenizer_modelscope_revision", "") or ""),
            "t5_tokenizer_modelscope_subfolder": str(getattr(args, "t5_tokenizer_modelscope_subfolder", "") or ""),
        }

    def build_resume_snapshot(self, args: argparse.Namespace) -> dict:
        return {
            "schema": "anima_resume_snapshot/v1",
            "fields": self._build_resume_snapshot_fields(args),
        }

    def validate_resume_snapshot(self, args: argparse.Namespace, snapshot: dict) -> None:
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid resume snapshot: expected a JSON object.")

        fields = snapshot.get("fields") if isinstance(snapshot.get("fields"), dict) else snapshot
        if not isinstance(fields, dict):
            raise ValueError("Invalid resume snapshot: missing fields object.")

        expected = self._build_resume_snapshot_fields(args)
        mismatches = []
        for key, expected_value in expected.items():
            if key not in fields:
                mismatches.append(f"{key}: missing in resume snapshot")
                continue
            saved_value = fields.get(key)
            if saved_value != expected_value:
                mismatches.append(f"{key}: resume={saved_value!r}, current={expected_value!r}")

        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(f"Resume snapshot mismatch detected. {details}")

    def _collect_memory_monitor_logs(self, accelerator: Accelerator) -> Dict[str, float]:
        args = self._runtime_args
        if args is None or not bool(getattr(args, "anima_monitor_memory", True)):
            return {}
        if accelerator.device.type != "cuda" or not torch.cuda.is_available():
            return {}

        device_index = accelerator.device.index if accelerator.device.index is not None else torch.cuda.current_device()
        bytes_in_mb = 1024.0 * 1024.0
        try:
            allocated_mb = float(torch.cuda.memory_allocated(device_index) / bytes_in_mb)
            reserved_mb = float(torch.cuda.memory_reserved(device_index) / bytes_in_mb)
            peak_allocated_mb = float(torch.cuda.max_memory_allocated(device_index) / bytes_in_mb)
            total_mb = float(torch.cuda.get_device_properties(device_index).total_memory / bytes_in_mb)
            usage_ratio = reserved_mb / total_mb if total_mb > 0 else 0.0
            warn_ratio = float(getattr(args, "anima_monitor_memory_warn_ratio", 0.95) or 0.95)
            near_limit = usage_ratio >= warn_ratio
            return {
                "gpu/mem_allocated_mb": allocated_mb,
                "gpu/mem_reserved_mb": reserved_mb,
                "gpu/mem_peak_allocated_mb": peak_allocated_mb,
                "gpu/mem_usage_ratio": usage_ratio,
                "alert/memory_near_limit": float(1.0 if near_limit else 0.0),
            }
        except Exception as exc:
            if not self._memory_metric_warning_logged:
                logger.warning("Failed to collect CUDA memory metrics: %s", exc)
                self._memory_metric_warning_logged = True
            return {"alert/memory_near_limit": 0.0}

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        transformer_path = str(getattr(args, "anima_transformer", "") or getattr(args, "pretrained_model_name_or_path", "") or "").strip()
        if not transformer_path:
            raise ValueError("Anima transformer path is required. Set --anima_transformer or --pretrained_model_name_or_path.")
        args.anima_transformer = transformer_path
        args.pretrained_model_name_or_path = transformer_path

        if not str(getattr(args, "vae", "") or "").strip():
            raise ValueError("Anima VAE path is required. Set --vae.")
        if not str(getattr(args, "qwen", "") or "").strip():
            raise ValueError("Qwen path is required. Set --qwen.")
        if not str(getattr(args, "t5_tokenizer_dir", "") or "").strip():
            raise ValueError("T5 tokenizer directory is required. Set --t5_tokenizer_dir.")

        if str(getattr(args, "network_module", "") or "").strip() == "":
            args.network_module = "networks.lora_anima"
        if args.network_module not in SUPPORTED_NETWORK_MODULES:
            raise ValueError(
                f"Unsupported network module: {args.network_module}. "
                f"Expected one of: {SUPPORTED_NETWORK_MODULES}"
            )

        if bool(getattr(args, "cache_text_encoder_outputs", False)) or bool(
            getattr(args, "cache_text_encoder_outputs_to_disk", False)
        ):
            raise ValueError("Anima native trainer currently does not support cache_text_encoder_outputs.")
        if not bool(getattr(args, "gradient_checkpointing", False)):
            logger.info("Anima native trainer forces gradient_checkpointing=true.")
        args.gradient_checkpointing = True

        optimizer_type = str(getattr(args, "optimizer_type", "") or "").strip().lower()
        if optimizer_type == "radam_schedulefree":
            args.optimizer_type = "RAdamScheduleFree"
            optimizer_type = "radamschedulefree"
        elif optimizer_type == "radamschedulefree":
            args.optimizer_type = "RAdamScheduleFree"

        if optimizer_type == "radamschedulefree":
            scheduler_name = str(getattr(args, "lr_scheduler", "") or "constant").strip().lower()
            if scheduler_name != "constant":
                raise ValueError("optimizer=RAdamScheduleFree requires lr_scheduler=constant")
            warmup_steps = float(getattr(args, "lr_warmup_steps", 0) or 0)
            if warmup_steps > 0:
                raise ValueError("optimizer=RAdamScheduleFree requires lr_warmup_steps=0")
            warmup_ratio = float(getattr(args, "lr_warmup_ratio", 0) or 0)
            if warmup_ratio > 0:
                raise ValueError("optimizer=RAdamScheduleFree requires lr_warmup_ratio=0")

        train_dataset_group.verify_bucket_reso_steps(int(getattr(args, "bucket_reso_steps", 64) or 64))
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(int(getattr(args, "bucket_reso_steps", 64) or 64))
        self._runtime_args = args

    def load_target_model(self, args, weight_dtype, accelerator):
        transformer = load_anima_transformer(
            args.anima_transformer,
            accelerator.device,
            weight_dtype,
            attention_backend=getattr(args, "anima_attention_backend", "torch"),
        )
        vae = load_anima_vae(args.vae, accelerator.device, weight_dtype)
        qwen_model = load_anima_qwen_model(args.qwen, accelerator.device, weight_dtype)
        return "anima-cosmos-predict2", [qwen_model], vae, transformer

    def get_tokenize_strategy(self, args):
        seq_len = int(getattr(args, "anima_seq_len", 512) or 512)
        self.qwen_tokenizer = load_anima_qwen_tokenizer(args.qwen)
        self.t5_tokenizer = load_anima_t5_tokenizer(
            args.t5_tokenizer_dir,
            auto_download=bool(getattr(args, "auto_download_t5_tokenizer", True)),
            repo_id=str(getattr(args, "t5_tokenizer_repo_id", "") or "google/t5-v1_1-base"),
            repo_subfolder=str(getattr(args, "t5_tokenizer_subfolder", "") or "tokenizer"),
            modelscope_fallback=bool(getattr(args, "t5_tokenizer_modelscope_fallback", True)),
            modelscope_repo_id=str(
                getattr(args, "t5_tokenizer_modelscope_repo_id", "")
                or "nv-community/Cosmos-Predict2-2B-Text2Image"
            ),
            modelscope_revision=str(getattr(args, "t5_tokenizer_modelscope_revision", "") or "master"),
            modelscope_subfolder=str(getattr(args, "t5_tokenizer_modelscope_subfolder", "") or "tokenizer"),
            strict_validation=bool(getattr(args, "t5_tokenizer_validate_strict", False)),
        )
        return AnimaTokenizeStrategy(
            qwen_tokenizer=self.qwen_tokenizer,
            t5_tokenizer=self.t5_tokenizer,
            max_length=seq_len,
        )

    def get_tokenizers(self, tokenize_strategy: AnimaTokenizeStrategy) -> List[Any]:
        return [tokenize_strategy.qwen_tokenizer, tokenize_strategy.t5_tokenizer]

    def get_latents_caching_strategy(self, args):
        return AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk,
            args.vae_batch_size,
            args.skip_cache_check,
        )

    def get_text_encoding_strategy(self, args):
        return AnimaTextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return text_encoders

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [False] * len(text_encoders)

    def is_train_text_encoder(self, args):
        return False

    def cast_text_encoder(self, args):
        return False

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        for model in text_encoders:
            model.eval()
            model.requires_grad_(False)
            model.to(accelerator.device)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        return DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )

    def _encode_images_with_vae(self, vae, images: torch.Tensor):
        images_5d = images.unsqueeze(2)
        if hasattr(vae, "model") and hasattr(vae, "scale"):
            latents = vae.model.encode(images_5d, vae.scale)
        elif hasattr(vae, "encode"):
            try:
                latents = vae.encode(images_5d, vae.scale)
            except TypeError:
                latents = vae.encode(images_5d)
        else:
            raise TypeError("Unsupported VAE object for Anima trainer.")
        return latents

    @staticmethod
    def _ensure_5d_latents(latents: torch.Tensor):
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        if latents.ndim != 5:
            raise ValueError(f"Expected 5D latents [B,C,T,H,W], got shape={tuple(latents.shape)}")
        return latents

    @staticmethod
    def _validation_timestep_if_fixed(args, batch_size: int, device: torch.device, dtype: torch.dtype):
        min_timestep = getattr(args, "min_timestep", None)
        max_timestep = getattr(args, "max_timestep", None)
        if min_timestep is None or max_timestep is None:
            return None
        if int(min_timestep) != int(max_timestep):
            return None
        t_value = float(min_timestep) / 1000.0
        return torch.full((batch_size,), t_value, device=device, dtype=dtype)

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy: strategy_base.TextEncodingStrategy,
        tokenize_strategy: strategy_base.TokenizeStrategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        del network, noise_scheduler, text_encoding_strategy, tokenize_strategy, train_text_encoder

        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = batch["latents"].to(accelerator.device)
            else:
                images = batch["images"].to(accelerator.device, dtype=vae_dtype)
                latents = self._encode_images_with_vae(vae, images)
            latents = self._ensure_5d_latents(latents).to(device=accelerator.device, dtype=weight_dtype)

        if "input_ids_list" not in batch or batch["input_ids_list"] is None:
            raise ValueError("Batch missing input_ids_list for Anima training.")
        input_ids_list = batch["input_ids_list"]
        if len(input_ids_list) < 3:
            raise ValueError("Anima training requires qwen_ids, qwen_attention_mask and t5_ids in input_ids_list.")

        qwen_input_ids = input_ids_list[0].to(accelerator.device, dtype=torch.long)
        qwen_attention_mask = input_ids_list[1].to(accelerator.device, dtype=torch.long)
        raw_t5_ids = input_ids_list[2]
        if raw_t5_ids is None:
            raise ValueError("T5 ids are required for Anima training.")
        t5_ids = raw_t5_ids.to(accelerator.device, dtype=torch.long)

        qwen_model = text_encoders[0]
        qwen_embeds = compute_qwen_embeddings(
            qwen_model,
            qwen_input_ids,
            qwen_attention_mask,
            no_grad=True,
        ).to(device=accelerator.device, dtype=weight_dtype)

        with torch.set_grad_enabled(is_train):
            cross = unet.preprocess_text_embeds(qwen_embeds, t5_ids).to(device=accelerator.device, dtype=weight_dtype)
        if cross.shape[1] < int(getattr(args, "anima_seq_len", 512)):
            target_len = int(getattr(args, "anima_seq_len", 512))
            pad_len = max(0, target_len - int(cross.shape[1]))
            if pad_len > 0:
                cross = torch.nn.functional.pad(cross, (0, 0, 0, pad_len))
        elif cross.shape[1] > int(getattr(args, "anima_seq_len", 512)):
            cross = cross[:, : int(getattr(args, "anima_seq_len", 512))]

        batch_size = int(latents.shape[0])
        t_values = self._validation_timestep_if_fixed(args, batch_size, accelerator.device, latents.dtype)
        if t_values is None:
            t_values = sample_t(
                batch_size,
                accelerator.device,
                method=str(getattr(args, "anima_timestep_sampling", "logit_normal")),
                sigmoid_scale=float(getattr(args, "anima_sigmoid_scale", 1.0)),
                shift=float(getattr(args, "anima_flow_shift", 3.0)),
            ).to(dtype=latents.dtype)

        noisy_latents, target = build_noisy_latents(
            latents,
            t_values,
            noise_offset=float(getattr(args, "anima_noise_offset", 0.0)),
        )
        padding_mask = torch.zeros(
            (batch_size, 1, noisy_latents.shape[-2], noisy_latents.shape[-1]),
            device=accelerator.device,
            dtype=noisy_latents.dtype,
        )

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            prediction = unet(
                noisy_latents.requires_grad_(train_unet),
                t_values,
                cross,
                padding_mask=padding_mask,
            )
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]

        reduction_dims = tuple(range(1, prediction.ndim))
        loss_per_sample = ((prediction.float() - target.float()) ** 2).mean(dim=reduction_dims)
        loss_weights = batch.get("loss_weights")
        if loss_weights is not None:
            loss_per_sample = loss_per_sample * loss_weights.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype)
        return loss_per_sample.mean()

    def generate_step_logs(
        self,
        args: argparse.Namespace,
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
        logs = super().generate_step_logs(
            args,
            current_loss,
            avr_loss,
            lr_scheduler,
            lr_descriptions,
            optimizer,
            keys_scaled,
            mean_norm,
            maximum_norm,
            mean_grad_norm,
            mean_combined_norm,
        )

        loss_spike_ratio = float(getattr(args, "anima_monitor_loss_spike_ratio", 3.0) or 3.0)
        nonfinite_loss = False
        loss_spike = False

        try:
            current_loss_value = float(current_loss)
            nonfinite_loss = not math.isfinite(current_loss_value)
            if nonfinite_loss:
                self._previous_loss_for_monitor = None
            else:
                prev = self._previous_loss_for_monitor
                if prev is not None and prev > 0 and current_loss_value > (prev * loss_spike_ratio):
                    loss_spike = True
                self._previous_loss_for_monitor = current_loss_value
        except Exception:
            nonfinite_loss = True
            self._previous_loss_for_monitor = None

        logs["alert/nonfinite_loss"] = float(1.0 if nonfinite_loss else 0.0)
        logs["alert/loss_spike"] = float(1.0 if loss_spike else 0.0)
        logs["alert/memory_near_limit"] = float(logs.get("alert/memory_near_limit", 0.0))
        logs["alert/any"] = float(1.0 if (nonfinite_loss or loss_spike) else 0.0)
        return logs

    def step_logging(self, accelerator: Accelerator, logs: dict, global_step: int, epoch: int):
        runtime_args = self._runtime_args
        merged_logs = dict(logs)
        merged_logs.update(self._collect_memory_monitor_logs(accelerator))

        nonfinite_loss = bool(float(merged_logs.get("alert/nonfinite_loss", 0.0)) > 0.0)
        loss_spike = bool(float(merged_logs.get("alert/loss_spike", 0.0)) > 0.0)
        memory_near_limit = bool(float(merged_logs.get("alert/memory_near_limit", 0.0)) > 0.0)
        merged_logs["alert/any"] = float(1.0 if (nonfinite_loss or loss_spike or memory_near_limit) else 0.0)

        super().step_logging(accelerator, merged_logs, global_step, epoch)

        if runtime_args is None:
            return

        alert_policy = str(getattr(runtime_args, "anima_monitor_alert_policy", "warn") or "warn").lower()
        if alert_policy == "raise" and (nonfinite_loss or memory_near_limit):
            reasons = []
            if nonfinite_loss:
                reasons.append("nonfinite_loss")
            if memory_near_limit:
                reasons.append("memory_near_limit")
            raise RuntimeError(
                f"Anima monitor alert policy=raise triggered at global_step={global_step}: {', '.join(reasons)}"
            )

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        del args, epoch, global_step, device, vae, tokenizer, text_encoder, unet
        if not self._sample_warned and accelerator.is_main_process:
            logger.info("Sample image generation is disabled in native anima_train_network for this phase.")
            self._sample_warned = True


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    parser.allow_abbrev = False
    parser.add_argument("--anima_transformer", type=str, default="", help="Path to Anima transformer safetensors.")
    parser.add_argument("--qwen", type=str, default="", help="Path to Qwen model directory.")
    parser.add_argument("--t5_tokenizer_dir", type=str, default="", help="Path to local T5 tokenizer directory.")
    parser.add_argument(
        "--auto_download_t5_tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-download required T5 tokenizer files when missing at startup.",
    )
    parser.add_argument(
        "--t5_tokenizer_repo_id",
        type=str,
        default="google/t5-v1_1-base",
        help="T5 tokenizer source (Hugging Face repo id/URL, or ModelScope URL) used for auto-download.",
    )
    parser.add_argument(
        "--t5_tokenizer_subfolder",
        type=str,
        default="tokenizer",
        help="Tokenizer subfolder inside Hugging Face repo (for Cosmos tokenizer layout).",
    )
    parser.add_argument(
        "--t5_tokenizer_modelscope_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback to ModelScope when Hugging Face T5 tokenizer download fails.",
    )
    parser.add_argument(
        "--t5_tokenizer_modelscope_repo_id",
        type=str,
        default="nv-community/Cosmos-Predict2-2B-Text2Image",
        help="ModelScope repo id used by T5 tokenizer fallback downloader.",
    )
    parser.add_argument(
        "--t5_tokenizer_modelscope_revision",
        type=str,
        default="master",
        help="ModelScope revision used by T5 tokenizer fallback downloader.",
    )
    parser.add_argument(
        "--t5_tokenizer_modelscope_subfolder",
        type=str,
        default="tokenizer",
        help="Tokenizer subfolder inside ModelScope fallback repo.",
    )
    parser.add_argument(
        "--t5_tokenizer_validate_strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable strict local tokenizer structure/id validation before training starts.",
    )
    parser.add_argument(
        "--anima_attention_backend",
        type=str,
        default="torch",
        choices=["torch", "xformers"],
        help="Attention backend for Anima transformer.",
    )
    parser.add_argument("--anima_seq_len", type=int, default=512, help="Text sequence length for Qwen/T5 tokenization.")
    parser.add_argument(
        "--anima_timestep_sampling",
        type=str,
        default="logit_normal",
        choices=["logit_normal", "uniform"],
        help="Timestep sampling strategy for Anima flow-style training.",
    )
    parser.add_argument("--anima_sigmoid_scale", type=float, default=1.0, help="Sigmoid scale for logit_normal timestep sampling.")
    parser.add_argument("--anima_flow_shift", type=float, default=3.0, help="Flow shift factor for timestep transform.")
    parser.add_argument("--anima_noise_offset", type=float, default=0.0, help="Optional channel-wise noise offset.")
    parser.add_argument(
        "--train_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train LayerNorm/RMSNorm weight/bias parameters alongside adapter layers.",
    )
    parser.add_argument(
        "--anima_monitor_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log CUDA memory metrics during training steps.",
    )
    parser.add_argument(
        "--anima_monitor_alert_policy",
        type=str,
        default="warn",
        choices=["warn", "raise"],
        help="Alert policy when monitor detects invalid loss or near-OOM memory usage.",
    )
    parser.add_argument(
        "--anima_monitor_memory_warn_ratio",
        type=float,
        default=0.95,
        help="Warn/raise threshold for reserved VRAM ratio (reserved / total).",
    )
    parser.add_argument(
        "--anima_monitor_loss_spike_ratio",
        type=float,
        default=3.0,
        help="Loss spike threshold ratio against previous finite step loss.",
    )
    parser.set_defaults(
        network_module="networks.lora_anima",
        caption_extension=".txt",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = normalize_optimizer_aliases(args)

    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    args = normalize_optimizer_aliases(args)
    args = apply_anima_network_defaults(args)

    trainer = AnimaNetworkTrainer()
    trainer.train(args)
