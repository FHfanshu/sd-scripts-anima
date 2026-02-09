from __future__ import annotations

import json
from pathlib import Path

import pytest

from library.anima_runtime import model_loading


def _write_min_t5_files(t5_dir: Path):
    t5_dir.mkdir(parents=True, exist_ok=True)
    (t5_dir / "config.json").write_text(json.dumps({"model_type": "t5", "vocab_size": 32128}), encoding="utf-8")
    (t5_dir / "spiece.model").write_bytes(b"fake-spm")
    (t5_dir / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_load_t5_tokenizer_auto_download_when_missing(tmp_path: Path, monkeypatch):
    target_dir = tmp_path / "missing_t5"
    calls = {"download": 0, "repo_id": None, "from_pretrained": 0}
    sentinel = object()

    def _fake_download(t5_dir: Path, repo_id: str, **kwargs):
        del kwargs
        calls["download"] += 1
        calls["repo_id"] = repo_id
        _write_min_t5_files(t5_dir)

    def _fake_from_pretrained(path: str, local_files_only: bool = False, **kwargs):
        del kwargs
        calls["from_pretrained"] += 1
        assert Path(path).resolve() == target_dir.resolve()
        assert local_files_only is True
        return sentinel

    monkeypatch.setattr(model_loading, "_download_t5_tokenizer_assets", _fake_download)
    monkeypatch.setattr("transformers.T5TokenizerFast.from_pretrained", _fake_from_pretrained)

    tokenizer = model_loading.load_t5_tokenizer(
        str(target_dir),
        auto_download=True,
        repo_id="google/t5-v1_1-base",
    )

    assert tokenizer is sentinel
    assert calls["download"] == 1
    assert calls["repo_id"] == "google/t5-v1_1-base"
    assert calls["from_pretrained"] == 1


def test_load_t5_tokenizer_no_auto_download_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="T5 tokenizer directory not found"):
        model_loading.load_t5_tokenizer(str(tmp_path / "missing_t5"), auto_download=False)


def test_download_t5_assets_prefers_hf(tmp_path: Path, monkeypatch):
    target_dir = tmp_path / "download_hf"
    calls = {"hf": 0, "ms": 0}

    def _fake_hf(t5_dir: Path, *, repo_id: str, revision: str, subfolder: str):
        calls["hf"] += 1
        assert repo_id == "nvidia/Cosmos-Predict2-2B-Text2Image"
        assert revision == "main"
        assert subfolder == "tokenizer"
        _write_min_t5_files(t5_dir)

    def _fake_ms(t5_dir: Path, *, repo_id: str, revision: str, subfolder: str):
        del t5_dir, repo_id, revision, subfolder
        calls["ms"] += 1

    monkeypatch.setattr(model_loading, "_download_t5_tokenizer_from_hf", _fake_hf)
    monkeypatch.setattr(model_loading, "_download_t5_tokenizer_from_modelscope", _fake_ms)

    model_loading._download_t5_tokenizer_assets(
        target_dir,
        repo_id="nvidia/Cosmos-Predict2-2B-Text2Image",
        repo_subfolder="tokenizer",
        modelscope_fallback=True,
    )
    assert calls["hf"] == 1
    assert calls["ms"] == 0


def test_download_t5_assets_fallback_to_modelscope_on_hf_error(tmp_path: Path, monkeypatch):
    target_dir = tmp_path / "download_ms_fallback"
    calls = {"hf": 0, "ms": 0}

    def _fake_hf(t5_dir: Path, *, repo_id: str, revision: str, subfolder: str):
        del t5_dir, repo_id, revision, subfolder
        calls["hf"] += 1
        raise RuntimeError("gated repo")

    def _fake_ms(t5_dir: Path, *, repo_id: str, revision: str, subfolder: str):
        calls["ms"] += 1
        assert repo_id == "nv-community/Cosmos-Predict2-2B-Text2Image"
        assert revision == "master"
        assert subfolder == "tokenizer"
        _write_min_t5_files(t5_dir)

    monkeypatch.setattr(model_loading, "_download_t5_tokenizer_from_hf", _fake_hf)
    monkeypatch.setattr(model_loading, "_download_t5_tokenizer_from_modelscope", _fake_ms)

    model_loading._download_t5_tokenizer_assets(
        target_dir,
        repo_id="nvidia/Cosmos-Predict2-2B-Text2Image",
        repo_subfolder="tokenizer",
        modelscope_fallback=True,
        modelscope_repo_id="nv-community/Cosmos-Predict2-2B-Text2Image",
        modelscope_revision="master",
        modelscope_subfolder="tokenizer",
    )
    assert calls["hf"] == 1
    assert calls["ms"] == 1


def test_parse_t5_source_hf_repo_id():
    source = model_loading._parse_t5_repo_source(
        "nvidia/Cosmos-Predict2-2B-Text2Image",
        default_provider="hf",
        default_revision="main",
        default_subfolder="tokenizer",
    )
    assert source.provider == "hf"
    assert source.repo_id == "nvidia/Cosmos-Predict2-2B-Text2Image"
    assert source.revision == "main"
    assert source.subfolder == "tokenizer"


def test_parse_t5_source_hf_tree_url():
    source = model_loading._parse_t5_repo_source(
        "https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image/tree/main/tokenizer",
        default_provider="hf",
        default_revision="main",
        default_subfolder="tokenizer",
    )
    assert source.provider == "hf"
    assert source.repo_id == "nvidia/Cosmos-Predict2-2B-Text2Image"
    assert source.revision == "main"
    assert source.subfolder == "tokenizer"


def test_parse_t5_source_modelscope_tree_url():
    source = model_loading._parse_t5_repo_source(
        "https://www.modelscope.cn/models/nv-community/Cosmos-Predict2-2B-Text2Image/tree/master/tokenizer",
        default_provider="modelscope",
        default_revision="master",
        default_subfolder="tokenizer",
    )
    assert source.provider == "modelscope"
    assert source.repo_id == "nv-community/Cosmos-Predict2-2B-Text2Image"
    assert source.revision == "master"
    assert source.subfolder == "tokenizer"
