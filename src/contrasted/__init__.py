"""Contrasted: supervised contrastive learning for protein domain classification."""

from contrasted.data import (
    EmbeddingStore,
    load_domain_ids_from_fasta,
    parse_fasta_header,
    read_fasta_sequences,
)
from contrasted.embed import EncodeConfig, ProstT5Encoder, encode_fasta
from contrasted.model import ContrastiveModel, ProjectionHead, project
from contrasted.search import VectorIndex
from contrasted.utils import get_device, load_labels

__version__ = "0.1.0"

__all__ = [
    "ContrastiveModel",
    "EmbeddingStore",
    "EncodeConfig",
    "ProjectionHead",
    "ProstT5Encoder",
    "VectorIndex",
    "encode_fasta",
    "get_device",
    "load_domain_ids_from_fasta",
    "load_labels",
    "parse_fasta_header",
    "project",
    "read_fasta_sequences",
]
