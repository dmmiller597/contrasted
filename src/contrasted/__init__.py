"""Contrasted: supervised contrastive learning for protein domain classification."""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"

__all__ = [
    "ConcatStoreRequest",
    "ContrastiveModel",
    "EmbeddingStore",
    "EncodeConfig",
    "ProjectionHead",
    "ProstT5Encoder",
    "VectorIndex",
    "as_centroid_index",
    "build_aa_3di_concat_store",
    "encode_fasta",
    "get_device",
    "load_domain_ids_from_fasta",
    "load_labels",
    "parse_fasta_header",
    "project",
    "read_fasta_sequences",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "EmbeddingStore": ("contrasted.data", "EmbeddingStore"),
    "load_domain_ids_from_fasta": ("contrasted.data", "load_domain_ids_from_fasta"),
    "parse_fasta_header": ("contrasted.data", "parse_fasta_header"),
    "read_fasta_sequences": ("contrasted.data", "read_fasta_sequences"),
    "EncodeConfig": ("contrasted.embed", "EncodeConfig"),
    "ProstT5Encoder": ("contrasted.embed", "ProstT5Encoder"),
    "encode_fasta": ("contrasted.embed", "encode_fasta"),
    "ConcatStoreRequest": ("contrasted.concat", "ConcatStoreRequest"),
    "build_aa_3di_concat_store": ("contrasted.concat", "build_aa_3di_concat_store"),
    "ContrastiveModel": ("contrasted.model", "ContrastiveModel"),
    "ProjectionHead": ("contrasted.model", "ProjectionHead"),
    "project": ("contrasted.model", "project"),
    "VectorIndex": ("contrasted.search", "VectorIndex"),
    "as_centroid_index": ("contrasted.search", "as_centroid_index"),
    "get_device": ("contrasted.utils", "get_device"),
    "load_labels": ("contrasted.utils", "load_labels"),
}


def __getattr__(name: str) -> Any:
    # Lazy exports keep ``import contrasted`` light until a symbol is used.
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
