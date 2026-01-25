"""PyTorch-based vector similarity search."""

import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _device_type(device: torch.device | str) -> str:
    return torch.device(device).type


def default_index_dtype(device: torch.device | str) -> torch.dtype:
    """Prefer float16 on GPU/MPS, float32 on CPU."""
    return torch.float16 if _device_type(device) in {"cuda", "mps"} else torch.float32


def _host_mem_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def default_query_chunk_size(
    n_db: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> int:
    """Choose a conservative chunk size based on device memory."""
    if n_db <= 0:
        return 1024

    bytes_per = torch.tensor([], dtype=dtype).element_size()
    device_type = _device_type(device)

    if device_type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        target_bytes = max(512 * 1024**2, int(total * 0.10))
        max_chunk = 65536
    else:
        total = _host_mem_bytes()
        target_bytes = int(total * 0.02) if total else 512 * 1024**2
        max_chunk = 8192

    chunk = max(1, target_bytes // (n_db * bytes_per))
    return int(min(max(chunk, 256), max_chunk))


class VectorIndex:
    """Vector index for k-NN search using PyTorch.

    Stores L2-normalized embeddings and performs cosine similarity search
    via matrix multiplication. Works on CPU, MPS, and CUDA.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        ids: list[str] | None = None,
        labels: list[str] | None = None,
        dtype: torch.dtype | None = None,
        *,
        normalized: bool = False,
    ):
        if dtype is None:
            dtype = default_index_dtype(embeddings.device)
        self.embeddings = embeddings.to(dtype)
        if not normalized:
            self.embeddings = self.embeddings / self.embeddings.norm(
                dim=1, keepdim=True
            ).clamp_min(1e-12)
        self.ids = ids
        self.labels = labels
        self.dtype = dtype

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1]

    def to(self, device: torch.device | str) -> "VectorIndex":
        self.embeddings = self.embeddings.to(device)
        return self

    def search(
        self,
        queries: torch.Tensor,
        k: int,
        *,
        chunk_size: int | None = None,
        normalize_queries: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Search for k nearest neighbors.

        Args:
            queries: (M, D) query embeddings (will be normalized)
            k: Number of neighbors
            chunk_size: Optional batch size for queries
            normalize_queries: Normalize queries before search

        Returns:
            similarities: (M, k) cosine similarities
            indices: (M, k) indices into self.ids
        """
        n_queries = queries.shape[0]
        n_db = self.embeddings.shape[0]

        if n_queries == 0 or n_db == 0:
            empty_scores = torch.empty(
                (n_queries, k), device=self.embeddings.device, dtype=self.dtype
            )
            empty_indices = torch.empty(
                (n_queries, k), device=self.embeddings.device, dtype=torch.long
            )
            return empty_scores, empty_indices

        k_eff = min(k, n_db)

        queries = queries.to(self.embeddings.device, self.dtype)
        if normalize_queries:
            queries = queries / queries.norm(dim=1, keepdim=True).clamp_min(1e-12)

        def _search_chunk(chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            similarities = chunk @ self.embeddings.T
            return similarities.topk(k_eff, dim=1)

        if chunk_size is None:
            chunk_size = default_query_chunk_size(
                n_db, self.embeddings.device, self.dtype
            )

        if n_queries <= chunk_size:
            scores, indices = _search_chunk(queries)
        else:
            scores_list = []
            indices_list = []
            for start in range(0, n_queries, chunk_size):
                end = start + chunk_size
                chunk_scores, chunk_indices = _search_chunk(queries[start:end])
                scores_list.append(chunk_scores)
                indices_list.append(chunk_indices)
            scores = torch.cat(scores_list, dim=0)
            indices = torch.cat(indices_list, dim=0)

        if k_eff < k:
            pad_scores = torch.zeros(
                (n_queries, k - k_eff),
                device=scores.device,
                dtype=scores.dtype,
            )
            pad_indices = torch.full(
                (n_queries, k - k_eff),
                -1,
                device=indices.device,
                dtype=indices.dtype,
            )
            scores = torch.cat([scores, pad_scores], dim=1)
            indices = torch.cat([indices, pad_indices], dim=1)

        return scores, indices

    def save(self, path: str | Path) -> None:
        """Save index to .pt file."""
        path = Path(path)
        torch.save(
            {
                "embeddings": self.embeddings.cpu(),
                "ids": self.ids,
                "labels": self.labels,
                "dtype": str(self.dtype),
            },
            path,
        )
        logger.info(f"Saved index ({len(self)} vectors) to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "VectorIndex":
        """Load index from .pt file."""
        path = Path(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        if dtype is None:
            dtype = default_index_dtype(device)
        index = cls(
            embeddings=data["embeddings"],
            ids=data["ids"],
            labels=data.get("labels"),
            dtype=dtype,
            normalized=True,
        )
        index.to(device)
        logger.info(f"Loaded index ({len(index)} vectors) from {path}")
        return index


def search_all(
    index: VectorIndex,
    queries: torch.Tensor,
    k: int,
    *,
    chunk_size: int | None = None,
    normalize_queries: bool = True,
):
    """Yield scores and indices for large query sets."""
    if queries.shape[0] == 0:
        return
    if chunk_size is None:
        chunk_size = default_query_chunk_size(
            index.embeddings.shape[0], index.embeddings.device, index.dtype
        )
    for start in range(0, queries.shape[0], chunk_size):
        end = start + chunk_size
        yield index.search(
            queries[start:end],
            k,
            chunk_size=None,
            normalize_queries=normalize_queries,
        )


def normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalize embeddings."""
    return embeddings / embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)


def normalize_numpy(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize numpy embeddings (float32)."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def search_numpy(
    queries: np.ndarray,
    database: np.ndarray,
    k: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Search normalized cosine similarity with numpy inputs."""
    queries_np = np.asarray(queries, dtype=np.float32)
    database_np = np.asarray(database, dtype=np.float32)

    n_queries = queries_np.shape[0]
    n_db = database_np.shape[0]
    if n_queries == 0 or n_db == 0:
        return (
            np.empty((n_queries, k), dtype=np.float32),
            np.empty((n_queries, k), dtype=np.int64),
        )

    k_eff = min(k, n_db)
    queries_t = torch.from_numpy(queries_np).to(device=device, dtype=dtype)
    database_t = torch.from_numpy(database_np).to(device=device, dtype=dtype)

    queries_t = queries_t / queries_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
    database_t = database_t / database_t.norm(dim=1, keepdim=True).clamp_min(1e-12)

    similarities = queries_t @ database_t.T
    scores, indices = similarities.topk(k_eff, dim=1)

    scores_np = scores.cpu().numpy().astype(np.float32)
    indices_np = indices.cpu().numpy().astype(np.int64)

    if k_eff < k:
        pad_scores = np.zeros((n_queries, k - k_eff), dtype=np.float32)
        pad_indices = -np.ones((n_queries, k - k_eff), dtype=np.int64)
        scores_np = np.concatenate([scores_np, pad_scores], axis=1)
        indices_np = np.concatenate([indices_np, pad_indices], axis=1)

    return scores_np, indices_np
