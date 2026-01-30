import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from contrasted.data import (
    EmbeddingDataModule,
    EmbeddingDataset,
    load_domain_ids_from_fasta,
    load_embedding_dir,
    parse_fasta_header,
)


def test_parse_fasta_header():
    header = ">cath|4_4_0|12e8H01/1-113"
    assert parse_fasta_header(header) == "12e8H01"


def test_parse_fasta_header_no_bracket():
    header = ">cath|4_4_0|12e8H01"
    assert parse_fasta_header(header) == "12e8H01"


def test_parse_fasta_header_ted():
    header = ">AF-A0A3N5ZM32-F1-model_v4_TED03"
    assert parse_fasta_header(header) == "AF-A0A3N5ZM32-F1-model_v4_TED03"


def test_parse_fasta_header_generic():
    header = ">foo_bar/1-100"
    assert parse_fasta_header(header) == "foo_bar"


def test_embedding_dataset():
    embeddings = torch.randn(100, 1024)
    labels = torch.randint(0, 10, (100,))
    indices = [0, 5, 10, 15]

    dataset = EmbeddingDataset(embeddings, labels, indices)

    assert len(dataset) == 4
    emb, label = dataset[0]
    assert emb.shape == (1024,)
    assert isinstance(label, int)


def test_load_domain_ids_from_fasta():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(">cath|4_4_0|12e8H01/1-113\n")
        f.write("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH\n")
        f.write(">cath|4_4_0|1a0aA00/1-100\n")
        f.write("MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKD\n")
        fasta_path = Path(f.name)

    domain_ids = load_domain_ids_from_fasta(fasta_path)
    assert domain_ids == ["12e8H01", "1a0aA00"]

    fasta_path.unlink()


def test_datamodule_setup():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        embeddings = np.random.randn(10, 1024).astype(np.float16)
        labels = np.arange(10, dtype=np.int64)
        ids = [f"dom{i:02d}" for i in range(10)]

        embedding_dir = tmpdir / "embeddings"
        embedding_dir.mkdir()
        np.save(embedding_dir / "embeddings.npy", embeddings)
        np.save(embedding_dir / "labels.npy", labels)
        (embedding_dir / "ids.txt").write_text("\n".join(ids) + "\n")
        metadata = {
            "dims": embeddings.shape[1],
            "count": embeddings.shape[0],
            "dtype": str(embeddings.dtype),
            "idx_to_label": {i: f"sf_{i}" for i in range(10)},
        }
        (embedding_dir / "metadata.json").write_text(json.dumps(metadata))

        train_fasta = tmpdir / "train.fasta"
        val_fasta = tmpdir / "val.fasta"
        test_fasta = tmpdir / "test.fasta"

        with open(train_fasta, "w") as f:
            for i in [0, 1, 2, 3, 4]:
                f.write(f">cath|4_4_0|dom{i:02d}/1-100\nSEQ\n")
        with open(val_fasta, "w") as f:
            for i in [5, 6]:
                f.write(f">cath|4_4_0|dom{i:02d}/1-100\nSEQ\n")
        with open(test_fasta, "w") as f:
            for i in [7, 8, 9]:
                f.write(f">cath|4_4_0|dom{i:02d}/1-100\nSEQ\n")

        dm = EmbeddingDataModule(
            train_fasta=str(train_fasta),
            val_fasta=str(val_fasta),
            test_fasta=str(test_fasta),
            embedding_dir=str(embedding_dir),
            batch_size=2,
            num_workers=0,
        )

        dm.setup("fit")

        assert len(dm.train_dataset) == 5
        assert len(dm.val_dataset) == 2
        assert dm.num_classes == 10


def test_load_embedding_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        embeddings = np.random.randn(4, 8).astype(np.float32)
        labels = np.array([0, 1, 0, 2], dtype=np.int64)
        ids = ["a", "b", "c", "d"]

        embedding_dir = tmpdir / "embeddings"
        embedding_dir.mkdir()
        np.save(embedding_dir / "embeddings.npy", embeddings)
        np.save(embedding_dir / "labels.npy", labels)
        (embedding_dir / "ids.txt").write_text("\n".join(ids) + "\n")
        metadata = {
            "dims": 8,
            "count": 4,
            "dtype": "float32",
        }
        (embedding_dir / "metadata.json").write_text(json.dumps(metadata))

        loaded_embeddings, loaded_ids, loaded_labels, idx_to_label = load_embedding_dir(
            embedding_dir
        )
        assert loaded_embeddings.shape == (4, 8)
        assert loaded_ids == ids
        assert np.array_equal(loaded_labels, labels)
        assert idx_to_label is None
