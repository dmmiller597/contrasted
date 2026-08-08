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
        labels: list[str | None] | None = None,
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
        chunk_size: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(similarities, indices)`` of the top ``k`` DB rows per query."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        n_db = len(self)
        # Empty query set: return well-shaped empty results (stable for callers).
        if queries.numel() == 0:
            device = self.embeddings.device
            return (
                torch.empty(queries.shape[0], k, device=device),
                torch.empty(queries.shape[0], k, dtype=torch.long, device=device),
            )
        # Non-empty queries against an empty index is a setup error: fail loudly
        # rather than returning uninitialized indices that read as garbage rows.
        if n_db == 0:
            raise ValueError("Cannot search an empty index (0 vectors).")

        queries = _normalize(queries.to(self.embeddings))
        k_eff = min(k, n_db)

        scores, indices = [], []
        for start in range(0, queries.shape[0], chunk_size):
            s, i = (queries[start : start + chunk_size] @ self.embeddings.T).topk(
                k_eff, dim=1
            )
            scores.append(s)
            indices.append(i)
        scores_t, indices_t = torch.cat(scores), torch.cat(indices)

        # The DB has fewer than k rows: pad each row out to width k with
        # zero scores and -1 sentinel indices so callers get a stable shape.
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
        # Index files hold only a tensor + lists of str/None, which load under
        # the safe weights_only=True path (no arbitrary-pickle execution).
        data = torch.load(path, map_location="cpu", weights_only=True)
        index = cls(
            data["embeddings"],
            ids=data.get("ids"),
            labels=data.get("labels"),
            normalized=True,
        )
        return index.to(device)


def as_centroid_index(index: VectorIndex) -> VectorIndex:
    """Collapse labeled rows to empirical H-centroids (L2-normalized means)."""
    if index.labels is None:
        raise ValueError("Centroid index requires labels.")
    if len(index.labels) != len(index):
        raise ValueError(
            "Centroid index requires one label per embedding row, "
            f"got {len(index.labels)} labels for {len(index)} rows."
        )

    labeled_rows = [
        row_index for row_index, label in enumerate(index.labels) if label is not None
    ]
    row_labels = [label for label in index.labels if label is not None]
    if not row_labels:
        raise ValueError("Centroid index requires at least one labeled row.")

    ordered_labels = sorted(set(row_labels))
    label_to_row = {label: row_index for row_index, label in enumerate(ordered_labels)}
    device = index.embeddings.device
    embedding_rows = torch.tensor(labeled_rows, dtype=torch.long, device=device)
    # MPS has no float64 kernels, so compute there on CPU and return to MPS.
    compute_device = torch.device("cpu") if device.type == "mps" else device
    centroid_rows = torch.tensor(
        [label_to_row[label] for label in row_labels],
        dtype=torch.long,
        device=compute_device,
    )

    values = index.embeddings.index_select(0, embedding_rows).to(compute_device)
    values = values.to(dtype=torch.float64)
    centroids = values.new_zeros((len(ordered_labels), index.dim))
    centroids.index_add_(0, centroid_rows, values)
    counts = torch.bincount(centroid_rows, minlength=len(ordered_labels)).to(centroids)
    centroids /= counts.unsqueeze(1)

    if torch.any(centroids.norm(dim=1) <= _EPS):
        raise ValueError("Cannot normalize a zero centroid.")
    centroids = _normalize(centroids)
    if device.type == "mps":
        centroids = centroids.to(device=device, dtype=index.embeddings.dtype)

    return VectorIndex(
        centroids,
        ids=ordered_labels,
        labels=ordered_labels,
        normalized=True,
    )
