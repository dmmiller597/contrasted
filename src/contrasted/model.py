"""Projection head and Lightning module for contrastive learning."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_HEAD_FORMAT = "contrasted_projection_head_v1"


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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)

    def save(self, path: str | Path) -> Path:
        """Persist the head as a small standalone artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": _HEAD_FORMAT,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "dropout": self.dropout,
                "state_dict": self.state_dict(),
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> ProjectionHead:
        """Load a head from a native artifact or a Lightning checkpoint.

        Native artifacts (written by :meth:`save`) carry the head dims directly.
        Lightning ``.ckpt`` files are detected by the presence of
        ``projection_head.net.*`` keys in their ``state_dict``; dims are
        recovered from the linear weight shapes.
        """
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(blob, dict) and blob.get("format") == _HEAD_FORMAT:
            head = cls(
                input_dim=int(blob["input_dim"]),
                hidden_dim=int(blob["hidden_dim"]),
                output_dim=int(blob["output_dim"]),
                dropout=float(blob.get("dropout", 0.0)),
            )
            head.load_state_dict(blob["state_dict"])
            head.eval()
            return head

        state_dict = blob.get("state_dict") if isinstance(blob, dict) else None
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unrecognized projection-head artifact: {path}")

        prefix = "projection_head."
        head_state = {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        if not head_state:
            raise ValueError(f"No projection_head.* tensors in checkpoint: {path}")

        first = head_state.get("net.0.weight")
        last = head_state.get("net.4.weight")
        if first is None or last is None:
            raise ValueError(f"Missing net.0.weight / net.4.weight in: {path}")
        hidden_dim, input_dim = first.shape
        output_dim, last_hidden = last.shape
        if hidden_dim != last_hidden:
            raise ValueError(
                f"Inconsistent hidden dims in {path}: {hidden_dim} vs {last_hidden}"
            )

        hparams = blob.get("hyper_parameters") if isinstance(blob, dict) else None
        dropout = 0.0
        if isinstance(hparams, dict):
            dropout = float(hparams.get("dropout", 0.0))

        head = cls(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            output_dim=int(output_dim),
            dropout=dropout,
        )
        head.load_state_dict(head_state)
        head.eval()
        return head


@torch.inference_mode()
def project(
    head: nn.Module,
    embeddings: np.ndarray | torch.Tensor,
    indices: Sequence[int] | None = None,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 4096,
    desc: str = "Projecting",
) -> torch.Tensor:
    """Run ``head`` over rows of ``embeddings`` (default: all rows) in chunks.

    Returns a CPU tensor of shape ``(len(rows), head.output_dim)``.
    """
    head.eval()
    rows: Sequence[int]
    if indices is None:
        rows = range(len(embeddings))
    else:
        rows = indices
    n = len(rows)
    if n == 0:
        return torch.empty(0, 0)

    chunks: list[torch.Tensor] = []
    for i in tqdm(range(0, n, batch_size), desc=desc, leave=False):
        sl = rows[i : i + batch_size]
        batch = embeddings[sl]
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(np.ascontiguousarray(batch))
        chunks.append(head(batch.float().to(device)).cpu())
    return torch.cat(chunks, dim=0)


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
