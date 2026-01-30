"""Contrasted: Supervised contrastive learning for protein domain classification."""

from contrasted.callbacks import KNNEvaluationCallback
from contrasted.data import (
    EmbeddingDataModule,
    EmbeddingDataset,
    load_embedding_dir,
    load_id_to_idx,
)
from contrasted.losses import ProxyAnchorLoss, SupConLoss
from contrasted.model import ContrastiveModel, ProjectionHead
from contrasted.search import FaissIndex, VectorIndex
from contrasted.utils import get_device, load_labels, set_seed

__version__ = "0.1.0"

__all__ = [
    "ContrastiveModel",
    "EmbeddingDataModule",
    "EmbeddingDataset",
    "FaissIndex",
    "KNNEvaluationCallback",
    "ProjectionHead",
    "ProxyAnchorLoss",
    "SupConLoss",
    "VectorIndex",
    "get_device",
    "load_embedding_dir",
    "load_id_to_idx",
    "load_labels",
    "set_seed",
]
