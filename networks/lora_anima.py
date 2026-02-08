from __future__ import annotations

import os

import torch

from ._anima_adapter_common import build_anima_network, infer_network_dim_from_weights


def create_network(
    multiplier: float,
    network_dim: int | None,
    network_alpha: float | None,
    ae,
    text_encoders,
    unet,
    neuron_dropout=None,
    **kwargs,
):
    return build_anima_network(
        "lora",
        multiplier,
        network_dim,
        network_alpha,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if file is None:
            raise ValueError("weights file must be provided when weights_sd is None")
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            obj = torch.load(file, map_location="cpu")
            weights_sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

    inferred_dim = infer_network_dim_from_weights("lora", weights_sd)
    network = build_anima_network(
        "lora",
        multiplier,
        kwargs.get("network_dim", inferred_dim),
        kwargs.get("network_alpha", float(inferred_dim)),
        **kwargs,
    )
    return network, weights_sd

