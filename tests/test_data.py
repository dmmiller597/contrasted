import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from contrasted.data import (
    EmbeddingDataModule,
    EmbeddingStore,
    load_domain_ids_from_fasta,
    parse_fasta_header,
)


@pytest.mark.parametrize(
    "header,expected",
    [
        (">cath|4_4_0|12e8H01/1-113", "12e8H01"),
        (">cath|4_4_0|12e8H01", "12e8H01"),
        (">AF-A0A3N5ZM32-F1-model_v4_TED03", "AF-A0A3N5ZM32-F1-model_v4_TED03"),
        (">foo_bar/1-100", "foo_bar"),
        # Non-CATH pipe-delimited headers must not hit the CATH branch: a
        # UniProt header keeps its full first token, not the third field.
        (">sp|P12345|NAME_HUMAN Description", "sp|P12345|NAME_HUMAN"),
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
        indices, found, missing = store.resolve(["dom02", "MISSING", "dom05", "dom00"])
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


def test_parse_fasta_header_bare_raises():
    # A bare ">" must raise ValueError (which readers catch) — not IndexError.
    with pytest.raises(ValueError):
        parse_fasta_header(">")


def test_load_domain_ids_skips_bare_header(tmp_path):
    fasta = tmp_path / "x.fa"
    fasta.write_text(">\nACDE\n>good_dom\nMKLV\n")
    assert load_domain_ids_from_fasta(fasta) == ["good_dom"]


def test_datamodule_fallback_covers_noncontiguous_labels(tmp_path):
    # Hand-built dir with non-contiguous labels and NO idx_to_label metadata:
    # the store's idx_to_label must span 0..max(label) for lookup.
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    ids = ["d0", "d1", "d2"]
    np.save(emb_dir / "embeddings.npy", np.random.randn(3, 8).astype(np.float32))
    np.save(emb_dir / "labels.npy", np.array([0, 5, 9], dtype=np.int64))
    (emb_dir / "ids.txt").write_text("\n".join(ids) + "\n")
    (emb_dir / "metadata.json").write_text(
        json.dumps({"dims": 8, "count": 3, "dtype": "float32"})  # no idx_to_label
    )
    fasta = tmp_path / "all.fasta"
    fasta.write_text("".join(f">{i}\nACDE\n" for i in ids))

    dm = EmbeddingDataModule(
        train_fasta=str(fasta),
        val_fasta=str(fasta),
        test_fasta=str(fasta),
        embedding_dir=str(emb_dir),
        num_workers=0,
    )
    dm._load_store()
    assert dm.store is not None
    assert dm.store.num_classes == 10  # max label 9 + 1, not 3 unique values


def test_datamodule_uses_train_classes(tmp_path):
    # num_classes is always the training split, not the store vocabulary.
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    ids = ["d0", "d1", "d2"]
    np.save(emb_dir / "embeddings.npy", np.random.randn(3, 8).astype(np.float32))
    np.save(emb_dir / "labels.npy", np.array([0, 5, 9], dtype=np.int64))
    (emb_dir / "ids.txt").write_text("\n".join(ids) + "\n")
    (emb_dir / "metadata.json").write_text(
        json.dumps({"dims": 8, "count": 3, "dtype": "float32"})
    )
    train = tmp_path / "train.fasta"
    train.write_text(">d0\nACDE\n>d1\nACDE\n")  # only 2 of 3 classes
    full = tmp_path / "all.fasta"
    full.write_text("".join(f">{i}\nACDE\n" for i in ids))

    dm = EmbeddingDataModule(
        train_fasta=str(train),
        val_fasta=str(train),
        test_fasta=str(train),
        embedding_dir=str(emb_dir),
        num_workers=0,
    )
    dm.setup("fit")
    assert dm.num_classes == 2  # two training classes, not 10 (store) or 3 (unique)

    dm2 = EmbeddingDataModule(
        train_fasta=str(train),
        val_fasta=str(full),
        test_fasta=str(train),
        embedding_dir=str(emb_dir),
        num_workers=0,
    )
    with pytest.raises(ValueError, match="absent from the training split"):
        dm2.setup("fit")


def test_datamodule_fallback_rejects_negative_labels(tmp_path):
    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    ids = ["d0", "d1"]
    np.save(emb_dir / "embeddings.npy", np.random.randn(2, 8).astype(np.float32))
    np.save(emb_dir / "labels.npy", np.array([-1, 0], dtype=np.int64))
    (emb_dir / "ids.txt").write_text("\n".join(ids) + "\n")
    (emb_dir / "metadata.json").write_text(
        json.dumps({"dims": 8, "count": 2, "dtype": "float32"})
    )
    fasta = tmp_path / "all.fasta"
    fasta.write_text("".join(f">{i}\nACDE\n" for i in ids))

    dm = EmbeddingDataModule(
        train_fasta=str(fasta),
        val_fasta=str(fasta),
        test_fasta=str(fasta),
        embedding_dir=str(emb_dir),
        num_workers=0,
    )
    with pytest.raises(ValueError, match="non-negative"):
        dm._load_store()
