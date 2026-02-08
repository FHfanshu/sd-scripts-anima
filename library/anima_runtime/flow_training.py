from __future__ import annotations

import contextlib

import torch


def _get_model_device(model):
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def compute_qwen_embeddings(qwen_model, input_ids, attention_mask, *, no_grad: bool = True):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    device = _get_model_device(qwen_model)
    input_ids = input_ids.to(device, dtype=torch.long)
    attention_mask = attention_mask.to(device, dtype=torch.long)
    grad_context = torch.no_grad() if no_grad else contextlib.nullcontext()
    with grad_context:
        outputs = qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = outputs.hidden_states[-1]
    lengths = attention_mask.sum(dim=1).to(device="cpu", dtype=torch.int64)
    seq_len = int(hidden_states.shape[1])
    for batch_index in range(hidden_states.shape[0]):
        length = int(lengths[batch_index].item())
        if length == 1:
            length = 0
        length = max(0, min(length, seq_len))
        hidden_states[batch_index, length:] = 0
    return hidden_states


def sample_t(
    batch_size: int,
    device: torch.device,
    *,
    method: str = "logit_normal",
    sigmoid_scale: float = 1.0,
    shift: float = 3.0,
):
    if method == "logit_normal":
        dist = torch.distributions.normal.Normal(0, 1)
    elif method == "uniform":
        dist = torch.distributions.uniform.Uniform(0, 1)
    else:
        raise ValueError(f"Unknown timestep sampling method: {method}")

    t_values = dist.sample((batch_size,)).to(device)
    if method == "logit_normal":
        t_values = torch.sigmoid(t_values * float(sigmoid_scale))
    if shift is not None:
        shift = float(shift)
        t_values = (t_values * shift) / (1 + (shift - 1) * t_values)
    return torch.clamp(t_values, min=0.0, max=1.0)


def build_noisy_latents(latents: torch.Tensor, t_values: torch.Tensor, noise_offset: float = 0.0):
    if latents.ndim != 5:
        raise ValueError(f"Expected 5D latents [B,C,T,H,W], got shape={tuple(latents.shape)}")
    if t_values.ndim != 1:
        raise ValueError(f"Expected 1D timesteps [B], got shape={tuple(t_values.shape)}")
    if t_values.shape[0] != latents.shape[0]:
        raise ValueError("Timesteps batch does not match latents batch size")

    noise = torch.randn_like(latents)
    if float(noise_offset) != 0.0:
        noise = noise + float(noise_offset) * torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1, 1),
            device=latents.device,
            dtype=latents.dtype,
        )
    t_view = t_values.view(-1, 1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    noisy_latents = (1.0 - t_view) * latents + t_view * noise
    target = noise - latents
    return noisy_latents, target
