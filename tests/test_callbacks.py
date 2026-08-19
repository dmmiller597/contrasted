"""Tests for Lightning callbacks."""

from __future__ import annotations

from unittest.mock import MagicMock

import lightning as L
import torch

from contrasted.callbacks import HeadExportCallback
from contrasted.losses import CenterContrastiveLoss
from contrasted.model import ContrastiveModel
from contrasted.projection import ProjectionHead


def _ccl(num_classes: int = 3, embedding_dim: int = 4) -> CenterContrastiveLoss:
    return CenterContrastiveLoss(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        margin=0.0,
        scale=16.0,
        lambda_=0.0,
        label_smoothing=0.0,
    )


def test_head_export_uses_best_checkpoint(tmp_path):
    """Exported head must match the best checkpoint, not the in-memory head."""
    torch.manual_seed(0)
    best_head = ProjectionHead(input_dim=8, hidden_dim=6, output_dim=4, dropout=0.0)
    torch.manual_seed(1)
    final_head = ProjectionHead(input_dim=8, hidden_dim=6, output_dim=4, dropout=0.0)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    best_ckpt = ckpt_dir / "best.ckpt"
    best_model = ContrastiveModel(
        projection_head=best_head,
        loss=_ccl(),
    )
    torch.save({"state_dict": best_model.state_dict()}, best_ckpt)

    model = ContrastiveModel(
        projection_head=final_head,
        loss=_ccl(),
    )

    ckpt_cb = MagicMock()
    ckpt_cb.dirpath = str(ckpt_dir)
    ckpt_cb.best_model_path = str(best_ckpt)

    trainer = MagicMock(spec=L.Trainer)
    trainer.checkpoint_callback = ckpt_cb
    trainer.default_root_dir = str(tmp_path / "lightning")

    HeadExportCallback().on_train_end(trainer, model)

    exported = ProjectionHead.load(ckpt_dir / "projection_head.pt")
    loaded_best = ProjectionHead.load(best_ckpt)
    for key, value in exported.state_dict().items():
        assert torch.equal(value, loaded_best.state_dict()[key])
