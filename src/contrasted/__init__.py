"""Contrasted: Supervised contrastive learning for protein domain classification."""

from contrasted.callbacks import KNNEvaluationCallback
from contrasted.data import EmbeddingDataModule, EmbeddingDataset
from contrasted.losses import ProxyAnchorLoss, SupConLoss
from contrasted.model import ContrastiveModel, ProjectionHead
from contrasted.search import VectorIndex
from contrasted.utils import load_labels, set_seed

__version__ = "0.1.0"

__all__ = [
    "EmbeddingDataModule",
    "EmbeddingDataset",
    "ContrastiveModel",
    "KNNEvaluationCallback",
    "ProjectionHead",
    "ProxyAnchorLoss",
    "SupConLoss",
    "VectorIndex",
    "load_labels",
    "set_seed",
]
