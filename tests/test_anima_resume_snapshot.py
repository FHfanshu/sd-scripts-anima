from __future__ import annotations

import copy

import pytest

from tests.anima_entry_test_utils import import_native_entry


def _build_args():
    native_entry = import_native_entry()
    parser = native_entry.setup_parser()
    args = parser.parse_args([])
    args.network_module = "networks.lokr_anima"
    args.network_dim = 16
    args.network_alpha = 16.0
    args.train_norm = True
    args.optimizer_type = "AdamW8bit"
    args.lr_scheduler = "constant"
    args.train_batch_size = 2
    args.gradient_accumulation_steps = 4
    args.mixed_precision = "bf16"
    args.anima_seq_len = 512
    args.t5_tokenizer_repo_id = "nvidia/Cosmos-Predict2-2B-Text2Image"
    args.t5_tokenizer_subfolder = "tokenizer"
    args.t5_tokenizer_modelscope_fallback = True
    args.t5_tokenizer_modelscope_repo_id = "nv-community/Cosmos-Predict2-2B-Text2Image"
    args.t5_tokenizer_modelscope_revision = "master"
    args.t5_tokenizer_modelscope_subfolder = "tokenizer"
    return native_entry, args


def test_build_resume_snapshot_contains_required_fields():
    native_entry, args = _build_args()
    trainer = native_entry.AnimaNetworkTrainer()
    snapshot = trainer.build_resume_snapshot(args)

    assert snapshot["schema"] == "anima_resume_snapshot/v1"
    fields = snapshot["fields"]
    assert fields["network_module"] == "networks.lokr_anima"
    assert fields["network_dim"] == 16
    assert fields["network_alpha"] == 16.0
    assert fields["train_norm"] is True
    assert fields["optimizer_type"] == "AdamW8bit"
    assert fields["lr_scheduler"] == "constant"
    assert fields["train_batch_size"] == 2
    assert fields["gradient_accumulation_steps"] == 4
    assert fields["mixed_precision"] == "bf16"
    assert fields["anima_seq_len"] == 512
    assert fields["t5_tokenizer_repo_id"] == "nvidia/Cosmos-Predict2-2B-Text2Image"


def test_validate_resume_snapshot_passes_when_matching():
    native_entry, args = _build_args()
    trainer = native_entry.AnimaNetworkTrainer()
    snapshot = trainer.build_resume_snapshot(args)
    trainer.validate_resume_snapshot(args, snapshot)


def test_validate_resume_snapshot_raises_on_mismatch():
    native_entry, args = _build_args()
    trainer = native_entry.AnimaNetworkTrainer()
    snapshot = trainer.build_resume_snapshot(args)
    tampered = copy.deepcopy(snapshot)
    tampered["fields"]["network_dim"] = 32

    with pytest.raises(ValueError, match="network_dim"):
        trainer.validate_resume_snapshot(args, tampered)
