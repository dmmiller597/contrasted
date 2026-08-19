"""Projection head and batched inference for ContrasTED.

Annotate and make_db import from here. The Lightning training module lives in
``contrasted.model``.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

_HEAD_FORMAT = "contrasted_projection_head_v1"


class ProjectionHead(nn.Module):
    """MLP projection head with L2-normalized outputs."""

    def __init__(
        self,
        input_dim: int = 2048,
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
        try:
            blob = torch.load(path, map_location=map_location, weights_only=True)
        except (pickle.UnpicklingError, TypeError) as error:
            logger.warning(
                "Weights-only loading failed for %s; retrying with "
                "weights_only=False for Lightning checkpoint compatibility: %s",
                path,
                error,
            )
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
    input_dim = getattr(head, "input_dim", None)
    if input_dim is not None and getattr(embeddings, "ndim", None) == 2:
        store_dim = int(embeddings.shape[1])
        if store_dim != int(input_dim):
            raise ValueError(
                f"Embedding store width {store_dim} does not match projection "
                f"head input_dim {input_dim}. Headline CCL AA∥3Di expects a "
                f"2048-d AA∥3Di store (contrasted-build-concat-store); "
                f"AA-only ProstT5 stores are 1024-d."
            )

    rows: Sequence[int] = range(len(embeddings)) if indices is None else indices
    n = len(rows)
    if n == 0:
        return torch.empty(0, getattr(head, "output_dim", 0))

    chunks: list[torch.Tensor] = []
    for i in tqdm(range(0, n, batch_size), desc=desc, leave=False):
        sl = rows[i : i + batch_size]
        batch = embeddings[sl]
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(np.ascontiguousarray(batch))
        chunks.append(head(batch.float().to(device)).cpu())
    return torch.cat(chunks, dim=0)
