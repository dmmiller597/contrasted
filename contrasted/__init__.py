"""Contrasted: Supervised contrastive learning for CATH protein superfamily classification."""

from contrasted.data import CathDataModule, CathEmbeddingDataset
from contrasted.model import CathSupConModel, ProjectionHead
from contrasted.losses import SupConLoss, ProxyAnchorLoss
from contrasted.utils import set_seed
from contrasted.callbacks import KNNEvaluationCallback

__version__ = "0.1.0"

__all__ = [
    "CathDataModule",
    "CathEmbeddingDataset",
    "CathSupConModel",
    "ProjectionHead",
    "SupConLoss",
    "ProxyAnchorLoss",
    "set_seed",
    "KNNEvaluationCallback",
]
