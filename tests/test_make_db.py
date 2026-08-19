"""Unit tests for make_db label resolution."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig

from contrasted.data import EmbeddingStore
from contrasted.make_db import resolve_labels, run
from contrasted.projection import ProjectionHead


def test_run_requires_embedding_dir():
    with pytest.raises(
        ValueError,
        match="embedding_dir is required for contrasted-make-db",
    ):
        run(DictConfig({"embedding_dir": None}))


def test_run_rejects_missing_store_ids(tmp_path: Path):
    store_dir = tmp_path / "store"
    store = EmbeddingStore(
        embeddings=np.zeros((1, 8), dtype=np.float32),
        ids=["present"],
        labels=None,
        id_to_idx={"present": 0},
        metadata={"modality": "aa_3di_concat"},
    )
    store.save(store_dir, extra_metadata={"modality": "aa_3di_concat"})

    fasta = tmp_path / "queries.fasta"
    fasta.write_text(">present\nAAAA\n>absent\nAAAA\n")

    head = ProjectionHead(input_dim=8, hidden_dim=8, output_dim=4, dropout=0.0)
    ckpt = tmp_path / "head.pt"
    head.save(ckpt)

    with pytest.raises(ValueError, match="1 domain IDs not found"):
        run(
            DictConfig(
                {
                    "embedding_dir": str(store_dir),
                    "input": str(fasta),
                    "model_path": str(ckpt),
                    "index_path": str(tmp_path / "index.pt"),
                    "ids": None,
                    "label_file": None,
                    "dtype": "float32",
                    "project_batch_size": 8,
                }
            )
        )


def test_resolve_labels_preserves_missing_as_none():
    id_to_label = {"a": 0, "b": 1, "c": 0}
    idx_to_label = {0: "SF_A", 1: "SF_B"}
    labels = resolve_labels(["a", "x", "b", "c"], id_to_label, idx_to_label)
    assert labels == ["SF_A", None, "SF_B", "SF_A"]


def test_resolve_labels_keeps_real_unknown_string():
    # A superfamily literally named "unknown" is a real label, not the sentinel.
    id_to_label = {"a": 0}
    idx_to_label = {0: "unknown"}
    labels = resolve_labels(["a", "missing"], id_to_label, idx_to_label)
    assert labels == ["unknown", None]
