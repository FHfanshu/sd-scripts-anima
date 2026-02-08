from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_anima_root() -> Path:
    env_root = os.environ.get("ANIMA_TRAINER_ROOT", "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path(__file__).resolve().parents[2])
    candidates.append(Path(__file__).resolve().parents[3])
    for candidate in candidates:
        if (candidate / "anima_train.py").exists() and (candidate / "utils" / "model_loading.py").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate Anima trainer root. Set ANIMA_TRAINER_ROOT to the directory containing anima_train.py."
    )


def _get_model_loading_module():
    root = _resolve_anima_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from utils import model_loading  # type: ignore

    return model_loading, root


def load_anima_transformer(ckpt_path, device, dtype, repo_root=None, attention_backend="torch"):
    model_loading, root = _get_model_loading_module()
    resolved_root = Path(repo_root) if repo_root else root
    return model_loading.load_anima_model(
        ckpt_path,
        device,
        dtype,
        resolved_root,
        attention_backend=attention_backend,
    )


def load_anima_vae(vae_path, device, dtype, repo_root=None):
    model_loading, root = _get_model_loading_module()
    resolved_root = Path(repo_root) if repo_root else root
    return model_loading.load_vae(vae_path, device, dtype, resolved_root)


def load_anima_qwen(qwen_path, device, dtype):
    model_loading, _ = _get_model_loading_module()
    return model_loading.load_qwen(qwen_path, device, dtype)


def load_anima_t5_tokenizer(t5_dir=None, repo_root=None):
    model_loading, root = _get_model_loading_module()
    resolved_root = Path(repo_root) if repo_root else root
    return model_loading.load_t5_tokenizer(resolved_root, t5_dir=t5_dir)

