"""PyTorch k-NN search over L2-normalized embeddings.

Brute-force cosine similarity via ``queries @ db.T`` + ``topk``. Runs on CPU,
MPS, or CUDA. For CATH-sized databases a flat inner-product search is as fast
as FAISS's own IndexFlatIP, with no extra dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(_EPS)


class VectorIndex:
    """L2-normalized embedding index with cosine-similarity k-NN search."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        ids: list[str] | None = None,
        labels: list[str] | None = None,
        *,
        normalized: bool = False,
    ):
        self.embeddings = embeddings if normalized else _normalize(embeddings)
        self.ids = ids
        self.labels = labels

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    def to(self, device: torch.device | str) -> VectorIndex:
        self.embeddings = self.embeddings.to(device)
        return self

    def search(
        self,
        queries: torch.Tensor,
        k: int,
        *,
        chunk_size: int | None = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(similarities, indices)`` of the top ``k`` DB rows per query."""
        n_db = len(self)
        if queries.numel() == 0 or n_db == 0:
            device = self.embeddings.device
            return (
                torch.empty(queries.shape[0], k, device=device),
                torch.empty(queries.shape[0], k, dtype=torch.long, device=device),
            )

        queries = _normalize(queries.to(self.embeddings))
        k_eff = min(k, n_db)
        step = chunk_size or queries.shape[0]

        scores, indices = [], []
        for start in range(0, queries.shape[0], step):
            s, i = (queries[start : start + step] @ self.embeddings.T).topk(
                k_eff, dim=1
            )
            scores.append(s)
            indices.append(i)
        scores_t, indices_t = torch.cat(scores), torch.cat(indices)

        if k_eff < k:
            pad_s = scores_t.new_zeros(queries.shape[0], k - k_eff)
            pad_i = indices_t.new_full((queries.shape[0], k - k_eff), -1)
            scores_t = torch.cat([scores_t, pad_s], dim=1)
            indices_t = torch.cat([indices_t, pad_i], dim=1)
        return scores_t, indices_t

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": self.embeddings.cpu(),
                "ids": self.ids,
                "labels": self.labels,
            },
            path,
        )
        logger.info(f"Saved index ({len(self)} vectors) to {path}")

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> VectorIndex:
        data = torch.load(path, map_location="cpu", weights_only=False)
        index = cls(
            data["embeddings"],
            ids=data.get("ids"),
            labels=data.get("labels"),
            normalized=True,
        )
        return index.to(device)
