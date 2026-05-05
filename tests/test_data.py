import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from contrasted.data import EmbeddingStore, parse_fasta_header


@pytest.mark.parametrize(
    "header,expected",
    [
        (">cath|4_4_0|12e8H01/1-113", "12e8H01"),
        (">cath|4_4_0|12e8H01", "12e8H01"),
        (">AF-A0A3N5ZM32-F1-model_v4_TED03", "AF-A0A3N5ZM32-F1-model_v4_TED03"),
        (">foo_bar/1-100", "foo_bar"),
    ],
)
def test_parse_fasta_header(header, expected):
    assert parse_fasta_header(header) == expected


def _make_embedding_dir(tmpdir: Path) -> Path:
    embeddings = np.random.randn(10, 8).astype(np.float32)
    ids = [f"dom{i:02d}" for i in range(10)]
    labels = np.arange(10, dtype=np.int64)
    embedding_dir = tmpdir / "embeddings"
    embedding_dir.mkdir()
    np.save(embedding_dir / "embeddings.npy", embeddings)
    np.save(embedding_dir / "labels.npy", labels)
    (embedding_dir / "ids.txt").write_text("\n".join(ids) + "\n")
    metadata = {
        "dims": 8,
        "count": 10,
        "dtype": "float32",
        "idx_to_label": {i: f"sf_{i}" for i in range(10)},
    }
    (embedding_dir / "metadata.json").write_text(json.dumps(metadata))
    return embedding_dir


def test_embedding_store_resolve():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EmbeddingStore.from_dir(_make_embedding_dir(Path(tmpdir)))
        indices, found, missing = store.resolve(
            ["dom02", "MISSING", "dom05", "dom00"]
        )
        assert indices == [2, 5, 0]
        assert found == ["dom02", "dom05", "dom00"]
        assert missing == ["MISSING"]


def test_embedding_store_get_batch():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EmbeddingStore.from_dir(_make_embedding_dir(Path(tmpdir)))
        batch, found, missing = store.get_batch(["dom01", "dom03", "nope"])
        assert batch.shape == (2, store.dim)
        assert batch.dtype == torch.float32
        assert found == ["dom01", "dom03"]
        assert missing == ["nope"]
