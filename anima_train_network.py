from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import logging

logger = logging.getLogger(__name__)

SUPPORTED_NETWORK_MODULES = ("networks.lora_anima", "networks.lokr_anima")


def _resolve_anima_train_py(explicit_path: str = "") -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"anima_train.py not found: {path}")

    env_path = os.environ.get("ANIMA_TRAIN_PY", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"ANIMA_TRAIN_PY points to non-existent file: {path}")

    candidates = [
        Path(__file__).resolve().parents[1] / "anima_train.py",
        Path(__file__).resolve().parents[2] / "anima_train.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate anima_train.py. Set --anima-train-py or ANIMA_TRAIN_PY."
    )


def _has_arg(args: List[str], key: str) -> bool:
    return any(item == key or item.startswith(key + "=") for item in args)


def build_forward_command(parsed: argparse.Namespace, passthrough: List[str]) -> List[str]:
    anima_train_py = _resolve_anima_train_py(parsed.anima_train_py)
    cmd = [sys.executable, str(anima_train_py)]
    if parsed.config:
        cmd.extend(["--config", parsed.config])

    if not _has_arg(passthrough, "--network-type"):
        network_type = "lokr" if parsed.network_module == "networks.lokr_anima" else "lora"
        cmd.extend(["--network-type", network_type])

    cmd.extend(passthrough)
    return cmd


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Anima network training entry for sd-scripts-anima.")
    parser.add_argument("--config", type=str, default="", help="Path to Anima TOML config.")
    parser.add_argument(
        "--network_module",
        type=str,
        default="networks.lora_anima",
        choices=list(SUPPORTED_NETWORK_MODULES),
        help="Adapter backend to use.",
    )
    parser.add_argument("--anima-train-py", type=str, default="", help="Explicit path to root anima_train.py.")
    parser.add_argument("--print-cmd", action="store_true", help="Print forwarded root command and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Alias of --print-cmd.")
    return parser.parse_known_args()


def main():
    args, passthrough = parse_args()
    cmd = build_forward_command(args, passthrough)
    logger.info("Forwarding to root trainer: %s", " ".join(cmd))

    if args.print_cmd or args.dry_run:
        print(" ".join(cmd))
        return

    proc = subprocess.run(cmd, check=False)
    raise SystemExit(int(proc.returncode))


if __name__ == "__main__":
    main()
