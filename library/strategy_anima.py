from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

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
    t5_input_ids: Optional[torch.Tensor]


class AnimaTokenizeStrategy(TokenizeStrategy):
    def __init__(self, qwen_tokenizer, t5_tokenizer=None, max_length: int = 512):
        self.qwen_tokenizer = qwen_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.max_length = int(max_length)

    def tokenize(self, text: str | List[str]) -> List[torch.Tensor]:
        texts = [text] if isinstance(text, str) else text
        qwen = self.qwen_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        t5_ids = None
        if self.t5_tokenizer is not None:
            t5 = self.t5_tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            t5_ids = t5["input_ids"]

        if t5_ids is None:
            return [qwen["input_ids"], qwen["attention_mask"]]
        return [qwen["input_ids"], qwen["attention_mask"], t5_ids]


class AnimaTextEncodingStrategy(TextEncodingStrategy):
    def encode_tokens(self, tokenize_strategy: AnimaTokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(models) < 2:
            raise ValueError("AnimaTextEncodingStrategy expects models=[qwen_model, anima_model]")
        qwen_model = models[0]
        anima_model = models[1]
        qwen_input_ids = tokens[0].to(next(qwen_model.parameters()).device)
        qwen_attention_mask = tokens[1].to(next(qwen_model.parameters()).device)
        t5_ids = tokens[2] if len(tokens) > 2 else None
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
                t5_ids.to(next(anima_model.parameters()).device) if t5_ids is not None else None,
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
        raise NotImplementedError("AnimaLatentsCachingStrategy.cache_batch_latents is provided by anima_train.py pipeline.")


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
            "AnimaTextEncoderOutputsCachingStrategy.cache_batch_outputs is provided by anima_train.py pipeline."
        )
