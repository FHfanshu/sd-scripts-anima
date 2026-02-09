from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from library import train_util


def test_lr_scheduler_num_cycles_accepts_float():
    parser = argparse.ArgumentParser()
    train_util.add_optimizer_arguments(parser)
    args = parser.parse_args(["--lr_scheduler_num_cycles", "0.5"])
    assert args.lr_scheduler_num_cycles == 0.5


def test_get_scheduler_fix_handles_float_cycles():
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    args = SimpleNamespace(
        optimizer_type="AdamW",
        lr_scheduler="cosine",
        max_train_steps=10,
        lr_warmup_steps=0,
        lr_decay_steps=0,
        lr_scheduler_num_cycles=0.5,
        lr_scheduler_power=1.0,
        lr_scheduler_timescale=None,
        lr_scheduler_min_lr_ratio=None,
        lr_scheduler_args=None,
        lr_scheduler_type="",
    )

    scheduler = train_util.get_scheduler_fix(args, optimizer, num_processes=1)
    assert scheduler is not None
