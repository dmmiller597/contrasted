"""Projection head and Lightning module for contrastive learning."""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import ProxyAnchorLoss, SupConLoss


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
    """Contrastive learning model for protein domain classification."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.0,
        loss_type: str = "supcon",
        loss_params: dict | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_classes: int | None = None,
        max_epochs: int = 200,
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

        self.projection_head = ProjectionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

        self._num_classes = num_classes
        self._loss_type = loss_type.lower()
        self._loss_params = loss_params or {}
        self._output_dim = output_dim

        if self._loss_type == "supcon":
            self.main_loss: SupConLoss | ProxyAnchorLoss = SupConLoss(
                temperature=self._loss_params.get("temperature", 0.07)
            )
        elif self._loss_type == "proxy_anchor":
            self.main_loss = None  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection_head(embeddings)

    def setup(self, stage: str):
        """Initialize components that require num_classes from datamodule."""
        if self._num_classes is None:
            if self.trainer.datamodule is None:  # type: ignore[union-attr]
                raise RuntimeError("Datamodule required when num_classes not specified")
            self._num_classes = self.trainer.datamodule.num_classes  # type: ignore[union-attr]

        if self._num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self._num_classes}")

        if self.main_loss is None:
            self.main_loss = ProxyAnchorLoss(
                num_classes=self._num_classes,
                embedding_dim=self._output_dim,
                margin=self._loss_params.get("margin", 0.1),
                alpha=self._loss_params.get("alpha", 32.0),
            )

    def _shared_step(self, batch, batch_idx, stage: str):
        embeddings, labels = batch
        projected = self(embeddings)

        assert self.main_loss is not None
        loss = self.main_loss(projected, labels)
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
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=self.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=self.min_lr,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
