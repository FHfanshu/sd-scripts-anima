from __future__ import annotations

from library.anima_runtime.model_loading import (
    load_anima_model,
    load_qwen,
    load_qwen_model,
    load_qwen_tokenizer,
    load_t5_tokenizer,
    load_vae,
)


def load_anima_transformer(ckpt_path, device, dtype, attention_backend="torch"):
    return load_anima_model(
        ckpt_path,
        device,
        dtype,
        attention_backend=attention_backend,
    )


def load_anima_vae(vae_path, device, dtype):
    return load_vae(vae_path, device, dtype)


def load_anima_qwen(qwen_path, device, dtype):
    return load_qwen(qwen_path, device, dtype)


def load_anima_qwen_tokenizer(qwen_path):
    return load_qwen_tokenizer(qwen_path)


def load_anima_qwen_model(qwen_path, device, dtype):
    return load_qwen_model(qwen_path, device, dtype)


def load_anima_t5_tokenizer(t5_dir):
    return load_t5_tokenizer(t5_dir)
