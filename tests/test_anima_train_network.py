from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anima_train_network as bridge


def test_build_forward_command_lora(monkeypatch):
    monkeypatch.setattr(bridge, "_resolve_anima_train_py", lambda _: Path("E:/repo/anima_train.py"))
    args = SimpleNamespace(
        config="config.toml",
        network_module="networks.lora_anima",
        anima_train_py="",
        print_cmd=False,
        dry_run=False,
    )
    cmd = bridge.build_forward_command(args, ["--epochs", "1"])
    assert cmd[1].endswith("anima_train.py")
    assert "--network-type" in cmd
    idx = cmd.index("--network-type")
    assert cmd[idx + 1] == "lora"


def test_build_forward_command_lokr_respects_passthrough(monkeypatch):
    monkeypatch.setattr(bridge, "_resolve_anima_train_py", lambda _: Path("E:/repo/anima_train.py"))
    args = SimpleNamespace(
        config="config.toml",
        network_module="networks.lokr_anima",
        anima_train_py="",
        print_cmd=False,
        dry_run=False,
    )
    cmd = bridge.build_forward_command(args, ["--network-type", "lora", "--epochs", "1"])
    assert cmd.count("--network-type") == 1
    idx = cmd.index("--network-type")
    assert cmd[idx + 1] == "lora"

