"""End-to-end training smoke test.

Runs 2 epochs on a tiny synthetic embedding store and asserts that training
completes, the KNN callback logs its metrics, and the loss actually decreases.
This is the single most effective regression test for the full wiring:
model + loss + data + callbacks + trainer.
"""

import json
from pathlib import Path

import lightning as L
import numpy as np
import torch

from contrasted.callbacks import KNNEvaluationCallback
from contrasted.data import EmbeddingDataModule
from contrasted.losses import SupConLoss
from contrasted.model import ContrastiveModel, ProjectionHead


def _make_synthetic_embedding_dir(
    tmpdir: Path,
    *,
    n_per_class: int = 50,
    num_classes: int = 4,
    dim: int = 32,
) -> Path:
    """Build a synthetic embedding directory with clustered Gaussians."""
    rng = np.random.default_rng(seed=0)
    centers = rng.normal(size=(num_classes, dim)).astype(np.float32)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    embs: list[np.ndarray] = []
    labels: list[int] = []
    ids: list[str] = []
    for c in range(num_classes):
        for i in range(n_per_class):
            vec = centers[c] + 0.15 * rng.normal(size=dim).astype(np.float32)
            embs.append(vec)
            labels.append(c)
            ids.append(f"dom_c{c:02d}_{i:03d}")

    embeddings = np.stack(embs, axis=0).astype(np.float32)
    labels_arr = np.asarray(labels, dtype=np.int64)

    embedding_dir = tmpdir / "embeddings"
    embedding_dir.mkdir()
    np.save(embedding_dir / "embeddings.npy", embeddings)
    np.save(embedding_dir / "labels.npy", labels_arr)
    (embedding_dir / "ids.txt").write_text("\n".join(ids) + "\n")
    (embedding_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dims": dim,
                "count": len(ids),
                "dtype": "float32",
                "idx_to_label": {i: f"sf_{i}" for i in range(num_classes)},
            }
        )
    )

    # Split: 70/15/15.
    rng.shuffle(ids)
    n_train = int(len(ids) * 0.7)
    n_val = int(len(ids) * 0.15)
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    def _write_fasta(path: Path, items: list[str]) -> None:
        with open(path, "w") as f:
            for i, d in enumerate(items):
                f.write(f">cath|4_4_0|{d}/1-100\nSEQ{i}\n")

    _write_fasta(tmpdir / "train.fasta", train_ids)
    _write_fasta(tmpdir / "val.fasta", val_ids)
    _write_fasta(tmpdir / "test.fasta", test_ids)

    return embedding_dir


def test_training_smoke(tmp_path):
    torch.manual_seed(0)
    embedding_dir = _make_synthetic_embedding_dir(tmp_path, dim=32)

    dm = EmbeddingDataModule(
        train_fasta=str(tmp_path / "train.fasta"),
        val_fasta=str(tmp_path / "val.fasta"),
        test_fasta=str(tmp_path / "test.fasta"),
        embedding_dir=str(embedding_dir),
        batch_size=32,
        num_workers=0,
        pin_memory=False,
    )

    head = ProjectionHead(input_dim=32, hidden_dim=32, output_dim=16, dropout=0.0)
    loss = SupConLoss(temperature=0.1)
    model = ContrastiveModel(
        projection_head=head,
        loss=loss,
        learning_rate=1e-2,
        weight_decay=0.0,
        max_epochs=2,
        warmup_epochs=0,
    )

    callback = KNNEvaluationCallback(eval_every_n_epochs=1)
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        callbacks=[callback],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=False,
    )
    trainer.fit(model, datamodule=dm)

    metrics = trainer.callback_metrics
    assert "train/loss" in metrics
    assert "val/loss" in metrics
    assert "val/knn_accuracy" in metrics
    # With well-separated clusters the knn accuracy should be high.
    assert float(metrics["val/knn_accuracy"]) >= 0.8
