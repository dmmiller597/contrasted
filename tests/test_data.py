import tempfile
from pathlib import Path

import torch

from contrasted.data import (
    EmbeddingDataModule,
    EmbeddingDataset,
    load_domain_ids_from_fasta,
    parse_fasta_header,
)


def test_parse_fasta_header():
    header = ">cath|4_4_0|12e8H01/1-113"
    assert parse_fasta_header(header) == "12e8H01"


def test_parse_fasta_header_no_bracket():
    header = ">cath|4_4_0|12e8H01"
    assert parse_fasta_header(header) == "12e8H01"


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

        embeddings = torch.randn(10, 1024)
        labels = torch.arange(10)
        ids = [f"dom{i:02d}" for i in range(10)]

        pt_path = tmpdir / "test.pt"
        torch.save(
            {
                "embeddings": embeddings,
                "labels": labels,
                "ids": ids,
                "idx_to_label": {i: f"sf_{i}" for i in range(10)},
            },
            pt_path,
        )

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
            embedding_file=str(pt_path),
            batch_size=2,
            num_workers=0,
        )

        dm.setup("fit")

        assert len(dm.train_dataset) == 5
        assert len(dm.val_dataset) == 2
        assert dm.num_classes == 10
