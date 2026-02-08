from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch

from tests.anima_entry_test_utils import import_native_entry


class _DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()


class _DummyQwenModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(128, hidden_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
        hidden = self.embed(input_ids)
        output = SimpleNamespace()
        output.hidden_states = [hidden, hidden]
        return output


class _DummyAnimaTransformer(torch.nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.last_t5_ids = None

    def preprocess_text_embeds(self, qwen_hidden, t5_ids):
        self.last_t5_ids = t5_ids.detach().clone()
        batch = qwen_hidden.shape[0]
        seq = t5_ids.shape[1]
        return torch.zeros((batch, seq, self.hidden_size), dtype=qwen_hidden.dtype, device=qwen_hidden.device)

    def forward(self, noisy_latents, t_values, cross, padding_mask=None):
        del t_values, cross, padding_mask
        return torch.zeros_like(noisy_latents)


def test_process_batch_passes_t5_ids_to_preprocess():
    native_entry = import_native_entry()
    trainer = native_entry.AnimaNetworkTrainer()
    accelerator = _DummyAccelerator()
    qwen_model = _DummyQwenModel()
    unet = _DummyAnimaTransformer()

    batch = {
        "latents": torch.randn(2, 16, 1, 4, 4),
        "input_ids_list": [
            torch.randint(0, 100, (2, 6), dtype=torch.long),
            torch.ones((2, 6), dtype=torch.long),
            torch.randint(0, 100, (2, 6), dtype=torch.long),
        ],
    }

    args = SimpleNamespace(
        anima_seq_len=6,
        min_timestep=None,
        max_timestep=None,
        anima_timestep_sampling="uniform",
        anima_sigmoid_scale=1.0,
        anima_flow_shift=1.0,
        anima_noise_offset=0.0,
    )

    loss = trainer.process_batch(
        batch=batch,
        text_encoders=[qwen_model],
        unet=unet,
        network=None,
        vae=None,
        noise_scheduler=None,
        vae_dtype=torch.float32,
        weight_dtype=torch.float32,
        accelerator=accelerator,
        args=args,
        text_encoding_strategy=None,
        tokenize_strategy=None,
        is_train=False,
        train_text_encoder=False,
        train_unet=False,
    )

    assert torch.isfinite(loss).item()
    assert unet.last_t5_ids is not None
    assert unet.last_t5_ids.shape[1] > 0
