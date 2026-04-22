"""Projection head and Lightning module for contrastive learning."""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head with L2-normalized outputs."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class ContrastiveModel(L.LightningModule):
    """Contrastive learning model for protein domain classification.

    The projection head and loss module are injected (typically via Hydra
    ``instantiate``) so the LightningModule itself is agnostic to which
    contrastive objective is in use.
    """

    def __init__(
        self,
        projection_head: nn.Module,
        loss: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["projection_head", "loss"])

        self.projection_head = projection_head
        self.loss = loss

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection_head(embeddings)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        embeddings, labels = batch
        projected = self(embeddings)
        loss = self.loss(projected, labels)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=hp["warmup_epochs"],
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hp["max_epochs"] - hp["warmup_epochs"],
            eta_min=hp["min_lr"],
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[hp["warmup_epochs"]],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
