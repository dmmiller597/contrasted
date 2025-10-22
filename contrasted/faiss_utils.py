"""FAISS utilities for efficient similarity search on CPU.

For L2-normalized embeddings, cosine similarity = inner product (IndexFlatIP).
All operations are CPU-optimized with proper type handling.
"""

import faiss
import numpy as np
import torch
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS IndexFlatIP for L2-normalized embeddings.
    
    Args:
        embeddings: (N, D) float32 array of L2-normalized embeddings
        
    Returns:
        faiss.Index: IndexFlatIP ready for cosine similarity search
        
    Raises:
        ValueError: If embeddings not float32 or not 2D
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
    
    n, d = embeddings.shape
    logger.debug(f"Building FAISS index: {n} embeddings × {d} dims")
    
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    return index


def search_faiss_index(
    index: faiss.Index,
    queries: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search FAISS index for k-nearest neighbors.
    
    Args:
        index: FAISS index built from normalized embeddings
        queries: (M, D) float32 array of normalized query embeddings
        k: Number of nearest neighbors
        
    Returns:
        similarities: (M, k) array of cosine similarities [0, 1]
        indices: (M, k) array of neighbor indices
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    similarities, indices = index.search(queries, k)
    return similarities, indices


def normalize_embeddings(embeddings: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to L2-normalized numpy array (float32).
    
    Args:
        embeddings: (N, D) torch.Tensor on any device
        
    Returns:
        (N, D) float32 numpy array, L2-normalized
    """
    # Move to CPU and convert to numpy
    emb_np = embeddings.cpu().detach().numpy().astype(np.float32)
    
    # L2 normalization (already typically done by model, but ensure it)
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    emb_np = np.divide(emb_np, norms + 1e-8)  # Add epsilon to avoid division by zero
    
    return emb_np
