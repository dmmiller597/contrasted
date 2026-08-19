"""End-to-end smoke tests for make_db.run and annotate.run on synthetic fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from contrasted.annotate import run as annotate_run
from contrasted.make_db import run as make_db_run
from contrasted.projection import ProjectionHead


def _write_embedding_dir(
    path: Path,
    *,
    ids: list[str],
    labels: list[int],
    dim: int = 8,
) -> None:
    path.mkdir()
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(ids), dim)).astype(np.float32)
    np.save(path / "embeddings.npy", embeddings)
    np.save(path / "labels.npy", np.asarray(labels, dtype=np.int64))
    (path / "ids.txt").write_text("\n".join(ids) + "\n")
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "dims": dim,
                "count": len(ids),
                "dtype": "float32",
                "idx_to_label": {i: f"SF_{i}" for i in sorted(set(labels))},
            }
        )
    )


def _write_label_file(path: Path, id_to_sf: dict[str, str]) -> None:
    lines = [f"{domain_id}\t{sf}" for domain_id, sf in id_to_sf.items()]
    path.write_text("\n".join(lines) + "\n")


def test_make_db_and_annotate_smoke(tmp_path):
    ids = ["ref_a", "ref_b", "query_a"]
    emb_dir = tmp_path / "embeddings"
    _write_embedding_dir(emb_dir, ids=ids, labels=[0, 1, 0])

    fasta = tmp_path / "domains.fasta"
    fasta.write_text(
        "".join(f">cath|1|{domain_id}/1-10\nACDEFGHIK\n" for domain_id in ids)
    )

    label_file = tmp_path / "labels.txt"
    _write_label_file(
        label_file,
        {"ref_a": "SF_0", "ref_b": "SF_1", "query_a": "SF_0"},
    )

    head = ProjectionHead(input_dim=8, hidden_dim=8, output_dim=4, dropout=0.0)
    head_path = tmp_path / "projection_head.pt"
    head.save(head_path)

    index_path = tmp_path / "index.pt"
    make_db_run(
        OmegaConf.create(
            {
                "input": str(fasta),
                "model_path": str(head_path),
                "index_path": str(index_path),
                "embedding_dir": str(emb_dir),
                "label_file": str(label_file),
                "ids": None,
            }
        )
    )
    assert index_path.is_file()

    out_dir = tmp_path / "annotations"
    annotate_run(
        OmegaConf.create(
            {
                "input": str(fasta),
                "model_path": str(head_path),
                "index": str(index_path),
                "embedding_dir": str(emb_dir),
                "id_to_annotation": str(label_file),
                "output_dir": str(out_dir),
                "k": 1,
                "distance_cutoff": 1.0,
                "compute_metrics": True,
                "return_true_annotation": True,
            }
        )
    )

    tsv = out_dir / "domains_annotations.tsv"
    metrics_path = out_dir / "domains_metrics.json"
    assert tsv.is_file()
    assert metrics_path.is_file()
    assert "query_id" in tsv.read_text()
    assert json.loads(metrics_path.read_text())["accuracy"] >= 0.0
