from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import os
import torch

from library.strategy_base import (
    LatentsCachingStrategy,
    TextEncoderOutputsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
)


@dataclass
class AnimaTokenBatch:
    qwen_input_ids: torch.Tensor
    qwen_attention_mask: torch.Tensor
    t5_input_ids: torch.Tensor


class AnimaTokenizeStrategy(TokenizeStrategy):
    def __init__(self, qwen_tokenizer, t5_tokenizer=None, max_length: int = 512):
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.max_length = int(max_length)

    def tokenize(self, text: str | List[str]) -> List[torch.Tensor]:
        if self.t5_tokenizer is None:
            raise ValueError("T5 tokenizer is required for Anima training")
        texts = [text] if isinstance(text, str) else text
        qwen = self.qwen_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        t5 = self.t5_tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        t5_ids = t5["input_ids"]
        return [qwen["input_ids"], qwen["attention_mask"], t5_ids]


class AnimaTextEncodingStrategy(TextEncodingStrategy):
    def encode_tokens(self, tokenize_strategy: AnimaTokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(models) < 2:
            raise ValueError("AnimaTextEncodingStrategy expects models=[qwen_model, anima_model]")
        if len(tokens) < 3:
            raise ValueError("AnimaTextEncodingStrategy requires [qwen_input_ids, qwen_attention_mask, t5_input_ids]")
        qwen_model = models[0]
        anima_model = models[1]
        qwen_input_ids = tokens[0].to(next(qwen_model.parameters()).device)
        qwen_attention_mask = tokens[1].to(next(qwen_model.parameters()).device)
        t5_ids = tokens[2]
        if t5_ids is None:
            raise ValueError("AnimaTextEncodingStrategy requires non-empty t5_input_ids.")
        with torch.no_grad():
            outputs = qwen_model(
                input_ids=qwen_input_ids,
                attention_mask=qwen_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]
            cross = anima_model.preprocess_text_embeds(
                hidden_states.to(dtype=next(anima_model.parameters()).dtype, device=next(anima_model.parameters()).device),
                t5_ids.to(next(anima_model.parameters()).device),
            )
        return [cross]


class AnimaLatentsCachingStrategy(LatentsCachingStrategy):
    def __init__(self, cache_to_disk: bool, batch_size: int, skip_cache_check: bool):
        super().__init__(cache_to_disk, batch_size, skip_cache_check)

    @property
    def cache_suffix(self) -> str:
        return "_anima.npz"

    def get_latents_npz_path(self, absolute_path: str, image_size: tuple[int, int]) -> str:
        stem, _ = os.path.splitext(absolute_path)
        return f"{stem}_{image_size[0]}x{image_size[1]}{self.cache_suffix}"

    def is_disk_cached_latents_expected(self, bucket_reso, npz_path: str, flip_aug: bool, alpha_mask: bool) -> bool:
        return os.path.exists(npz_path)

    def cache_batch_latents(self, model: Any, batch: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        from library import train_util

        if not hasattr(model, "model") or not hasattr(model, "scale"):
            raise TypeError("Anima VAE handle must provide model/scale for latents caching")

        img_tensor, alpha_masks, original_sizes, crop_ltrbs = train_util.load_images_and_masks_for_caching(
            batch, alpha_mask, random_crop
        )
        img_tensor = img_tensor.to(device=model.device, dtype=model.dtype)

        with torch.no_grad():
            latents_tensors = model.model.encode(img_tensor.unsqueeze(2), model.scale).to("cpu")
        if latents_tensors.ndim == 5 and latents_tensors.shape[2] == 1:
            latents_tensors = latents_tensors.squeeze(2)

        if flip_aug:
            flipped = torch.flip(img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = model.model.encode(flipped.unsqueeze(2), model.scale).to("cpu")
            if flipped_latents.ndim == 5 and flipped_latents.shape[2] == 1:
                flipped_latents = flipped_latents.squeeze(2)
        else:
            flipped_latents = [None] * len(latents_tensors)

        for index, info in enumerate(batch):
            latents = latents_tensors[index]
            flipped_latent = flipped_latents[index]
            alpha_mask_tensor = alpha_masks[index]
            original_size = original_sizes[index]
            crop_ltrb = crop_ltrbs[index]

            if self.cache_to_disk:
                self.save_latents_to_disk(
                    info.latents_npz,
                    latents,
                    original_size,
                    crop_ltrb,
                    flipped_latent,
                    alpha_mask_tensor,
                )
            else:
                info.latents_original_size = original_size
                info.latents_crop_ltrb = crop_ltrb
                info.latents = latents
                if flip_aug:
                    info.latents_flipped = flipped_latent
                info.alpha_mask = alpha_mask_tensor

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(model.device)


class AnimaTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    def __init__(self, cache_to_disk: bool, batch_size: int, skip_cache_check: bool):
        super().__init__(cache_to_disk, batch_size, skip_cache_check)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return image_abs_path + "_anima_te.npz"

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        return [data[k] for k in data.files]

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        return os.path.exists(npz_path)

    def cache_batch_outputs(self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, batch: List):
        raise NotImplementedError(
            "AnimaTextEncoderOutputsCachingStrategy is not enabled in native anima_train_network yet."
        )
