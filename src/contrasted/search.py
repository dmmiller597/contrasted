"""PyTorch-based vector similarity search."""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Workaround for OpenMP conflicts between FAISS and PyTorch on ARM Macs.
# See: https://github.com/microsoft/LightGBM/issues/6595
if sys.platform == "darwin" and "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]

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


class FaissIndex:
    """FAISS-based vector index using cosine similarity (inner product)."""

    def __init__(
        self,
        embeddings: torch.Tensor | np.ndarray,
        ids: list[str] | None = None,
        labels: list[str] | None = None,
        *,
        normalized: bool = False,
    ):
        if faiss is None:
            raise ImportError("faiss-cpu is required for FaissIndex")
        embeddings_np = self._to_numpy(embeddings)
        if not normalized:
            embeddings_np = normalize_numpy(embeddings_np)
        self.index = faiss.IndexFlatIP(embeddings_np.shape[1])
        self.index.add(embeddings_np)
        self.ids = ids
        self.labels = labels

    @staticmethod
    def _to_numpy(embeddings: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.index.ntotal)

    @property
    def dim(self) -> int:
        return int(self.index.d)

    def to(self, device: torch.device | str) -> "FaissIndex":
        """No-op for API compatibility. FAISS indices run on CPU only."""
        return self

    def search(
        self,
        queries: torch.Tensor,
        k: int,
        *,
        chunk_size: int | None = None,
        normalize_queries: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if faiss is None:
            raise ImportError("faiss-cpu is required for FaissIndex")
        if queries.numel() == 0 or len(self) == 0:
            empty_scores = torch.empty((queries.shape[0], k), dtype=torch.float32)
            empty_indices = torch.empty((queries.shape[0], k), dtype=torch.long)
            return empty_scores, empty_indices

        queries_np = self._to_numpy(queries)
        if normalize_queries:
            queries_np = normalize_numpy(queries_np)

        k_eff = min(k, len(self))
        scores_np, indices_np = self.index.search(queries_np, k_eff)

        if k_eff < k:
            pad_scores = np.zeros((queries_np.shape[0], k - k_eff), dtype=np.float32)
            pad_indices = -np.ones((queries_np.shape[0], k - k_eff), dtype=np.int64)
            scores_np = np.concatenate([scores_np, pad_scores], axis=1)
            indices_np = np.concatenate([indices_np, pad_indices], axis=1)

        scores = torch.from_numpy(scores_np)
        indices = torch.from_numpy(indices_np)
        return scores, indices

    def save(self, path: str | Path) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu is required for FaissIndex")
        path = Path(path)
        payload = {
            "faiss_index": faiss.serialize_index(self.index),
            "ids": self.ids,
            "labels": self.labels,
            "dim": self.dim,
        }
        torch.save(payload, path)
        logger.info(f"Saved FAISS index ({len(self)} vectors) to {path}")

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "FaissIndex":
        """Load index from .pt file.

        Note: device parameter is accepted for API compatibility with VectorIndex
        but FAISS indices always run on CPU. Use VectorIndex for GPU acceleration.
        """
        if faiss is None:
            raise ImportError("faiss-cpu is required for FaissIndex")
        path = Path(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        index = faiss.deserialize_index(data["faiss_index"])
        instance = cls.__new__(cls)
        instance.index = index
        instance.ids = data.get("ids")
        instance.labels = data.get("labels")
        return instance


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
    *,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """k-NN search for numpy arrays with torch backend.

    Returns cosine similarities and indices into the database.
    """
    if device is None:
        device = "cpu"
    device = torch.device(device)

    if queries.size == 0 or database.size == 0:
        return (
            np.empty((len(queries), k), dtype=np.float32),
            np.empty((len(queries), k), dtype=np.int64),
        )

    db_tensor = torch.from_numpy(database).to(device)
    query_tensor = torch.from_numpy(queries).to(device)

    index = VectorIndex(db_tensor, normalized=True)
    scores, indices = index.search(query_tensor, k)
    return scores.cpu().numpy(), indices.cpu().numpy()
