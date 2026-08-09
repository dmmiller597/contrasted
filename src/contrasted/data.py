"""Embedding store I/O and FASTA helpers for ContrasTED.

Training uses ``contrasted.datamodule``. Annotate / make_db / embed / concat
use this module.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from contrasted.utils import require_exists

logger = logging.getLogger(__name__)

CANONICAL_EMBEDDING_FILES = (
    "embeddings.npy",
    "labels.npy",
    "ids.txt",
    "metadata.json",
)

def parse_fasta_header(header: str) -> str:
    """Extract a domain/sequence ID from a FASTA header line.

    Recognized formats:
    - CATH: ``>cath|{cath_release}|{domain_id}/{start}-{end}`` -> ``domain_id``
    - Fallback (including TED/AlphaFold ``>AF-..._TED03``): first whitespace
      token after ``>``, with any trailing ``/start-end`` stripped.
    """
    header = header.strip()
    if not header.startswith(">"):
        raise ValueError(f"Invalid FASTA header: {header}")

    tokens = header[1:].split()
    if not tokens:
        # A bare ">" (no id). Raise ValueError so callers skip/warn consistently
        # rather than crashing on an uncaught IndexError.
        raise ValueError(f"Empty FASTA header: {header!r}")
    token = tokens[0]
    parts = token.split("|")
    if parts[0] == "cath" and len(parts) >= 3:
        return parts[2].split("/", 1)[0]
    return token.split("/", 1)[0]


def load_domain_ids_from_fasta(fasta_path: str | Path) -> list[str]:
    """Read a FASTA file and return the list of parsed domain IDs (in order)."""
    domain_ids: list[str] = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                try:
                    domain_ids.append(parse_fasta_header(line))
                except ValueError as e:
                    logger.warning(f"Could not parse header: {line.strip()} - {e}")
    return domain_ids


def read_fasta_sequences(fasta_path: str | Path) -> dict[str, str]:
    """Read a FASTA file into an ordered ``{domain_id: sequence}`` map.

    Gaps (``-``) are stripped and multi-line sequences are joined. Duplicate
    IDs keep the first occurrence and emit a warning.
    """
    # Accumulate residue lines per id and join once, rather than repeated
    # string concatenation (quadratic for long, many-line records).
    chunks: dict[str, list[str]] = {}
    current_id: str | None = None
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                try:
                    current_id = parse_fasta_header(line)
                except ValueError as e:
                    logger.warning(f"Skipping invalid header: {e}")
                    current_id = None
                    continue
                if current_id in chunks:
                    logger.warning(f"Duplicate ID '{current_id}'; keeping first")
                    current_id = None
                else:
                    chunks[current_id] = []
            elif current_id is not None:
                chunks[current_id].append(line.replace("-", ""))
    sequences = {did: "".join(parts) for did, parts in chunks.items()}
    for did, seq in sequences.items():
        if not seq:
            raise ValueError(
                f"FASTA record '{did}' has no residues "
                f"(header-only or gap-only sequence in {fasta_path})"
            )
    return sequences


def resolve_fasta_paths(fasta_input: str | Path) -> dict[str, Path]:
    """Resolve ``fasta_input`` to a dict mapping split name -> path.

    Accepts a single file or a directory; returns ``{}`` if nothing is found.
    """
    fasta_input = Path(fasta_input)
    if fasta_input.is_dir():
        fasta_files = sorted(fasta_input.glob("*.fasta")) + sorted(
            fasta_input.glob("*.fa")
        )
        if not fasta_files:
            logger.warning(f"No FASTA files found in directory: {fasta_input}")
            return {}
        logger.info(f"Found {len(fasta_files)} FASTA files in {fasta_input}")
        return {f.stem: f for f in fasta_files}
    if fasta_input.is_file():
        return {fasta_input.stem: fasta_input}
    logger.warning(f"FASTA path not found: {fasta_input}")
    return {}


# ---------------------------------------------------------------------------
# Embedding directory loading
# ---------------------------------------------------------------------------


def read_ids_txt(path: Path) -> list[str]:
    """Domain ids from a store's ``ids.txt``, in row order.

    Line *i* is authoritative for ``embeddings.npy`` row *i*, so this is part of
    the store format contract rather than an internal helper.
    """
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _load_idx_to_label(embedding_dir: Path, metadata: dict) -> dict[int, str] | None:
    if idx_to_label := metadata.get("idx_to_label"):
        if isinstance(idx_to_label, dict):
            return {int(k): v for k, v in idx_to_label.items()}

    if (sidecar := metadata.get("idx_to_label_file")) and (
        path := embedding_dir / sidecar
    ).exists():
        with open(path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return None


# ---------------------------------------------------------------------------
# EmbeddingStore: the single entry point for embedding-directory I/O
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingStore:
    """Owns the contents of an embedding directory.

    The directory must contain:
    - ``embeddings.npy``: ``(N, D)`` array (memory-mapped on load)
    - ``ids.txt``: one ID per line
    - ``metadata.json``: at minimum ``dims``/``count``/``dtype``
    - ``labels.npy`` (optional): ``(N,)`` int64 class indices

    The ``ids.txt`` line order is authoritative: row ``i`` of ``embeddings.npy``
    is domain ``ids[i]``, so the ID -> row mapping is rebuilt from ``ids.txt``.
    """

    embeddings: np.ndarray
    ids: list[str]
    labels: np.ndarray | None
    id_to_idx: dict[str, int]
    idx_to_label: dict[int, str] | None = None
    embedding_dir: Path | None = field(default=None, compare=False)
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        *,
        require_labels: bool = False,
    ) -> "EmbeddingStore":
        path = Path(path)
        embeddings_path = path / "embeddings.npy"
        ids_path = path / "ids.txt"
        metadata_path = path / "metadata.json"
        labels_path = path / "labels.npy"

        for required, name in [
            (embeddings_path, "embeddings.npy"),
            (ids_path, "ids.txt"),
            (metadata_path, "metadata.json"),
        ]:
            require_exists(required, name)

        with open(metadata_path) as f:
            metadata = json.load(f)

        embeddings = np.load(embeddings_path, mmap_mode="r")

        for key, actual, cast in [
            ("dtype", embeddings.dtype, np.dtype),
            ("dims", embeddings.shape[1], int),
            ("count", embeddings.shape[0], int),
        ]:
            if (expected := metadata.get(key)) is not None and actual != cast(expected):
                raise ValueError(
                    f"embeddings.npy {key} {actual} != metadata {key} {expected}"
                )

        ids = read_ids_txt(ids_path)
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"ids.txt has {len(ids)} entries but embeddings.npy has "
                f"{embeddings.shape[0]}"
            )

        labels: np.ndarray | None = None
        if labels_path.exists():
            labels = np.asarray(np.load(labels_path), dtype=np.int64)
            if labels.shape[0] != embeddings.shape[0]:
                raise ValueError(
                    f"labels.npy has {labels.shape[0]} entries but "
                    f"embeddings.npy has {embeddings.shape[0]}"
                )
        elif require_labels:
            raise FileNotFoundError(f"Missing labels.npy in {path}")

        idx_to_label = _load_idx_to_label(path, metadata)

        return cls(
            embeddings=embeddings,
            ids=ids,
            labels=labels,
            id_to_idx={id_: i for i, id_ in enumerate(ids)},
            idx_to_label=idx_to_label,
            embedding_dir=path,
            metadata=metadata,
        )

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_label) if self.idx_to_label else 0

    def resolve(self, domain_ids: list[str]) -> tuple[list[int], list[str], list[str]]:
        """Map domain IDs to embedding-row indices.

        Returns ``(found_indices, found_ids, missing_ids)`` in the same
        order as ``domain_ids``.
        """
        found_idx, found_ids, missing_ids = [], [], []
        for d in domain_ids:
            if (row := self.id_to_idx.get(d)) is not None:
                found_idx.append(row)
                found_ids.append(d)
            else:
                missing_ids.append(d)
        return found_idx, found_ids, missing_ids

    def get_batch(
        self, domain_ids: list[str]
    ) -> tuple[torch.Tensor, list[str], list[str]]:
        """Fetch embeddings for ``domain_ids`` as a ``(M, D)`` float tensor."""
        found_idx, found_ids, missing_ids = self.resolve(domain_ids)
        batch = (
            torch.from_numpy(np.array(self.embeddings[found_idx], copy=True)).float()
            if found_idx
            else torch.empty(0, self.dim)
        )
        return batch, found_ids, missing_ids

    def save(
        self,
        path: str | Path,
        *,
        source: str | None = None,
        extra_metadata: dict | None = None,
    ) -> Path:
        """Write the canonical 4-file layout to ``path``.

        Writes are atomic per-file: each file is staged as ``<name>.tmp`` in
        the target directory and renamed on success. ``path`` must not
        already contain any of the canonical files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        existing = [n for n in CANONICAL_EMBEDDING_FILES if (path / n).exists()]
        if existing:
            raise FileExistsError(
                f"{path} already contains {existing}; refusing to overwrite"
            )

        def _atomic(name: str, writer):
            tmp = path / f"{name}.tmp"
            writer(tmp)
            tmp.replace(path / name)

        def _save_npy(path: Path, value) -> None:
            with path.open("wb") as f:
                np.save(f, value)

        embeddings = np.ascontiguousarray(self.embeddings)
        _atomic("embeddings.npy", lambda p: _save_npy(p, embeddings))

        if self.labels is not None:
            _atomic("labels.npy", lambda p: _save_npy(p, np.asarray(self.labels)))

        _atomic(
            "ids.txt",
            lambda p: p.write_text("\n".join(self.ids) + "\n"),
        )

        metadata = {
            "dims": int(embeddings.shape[1]),
            "count": int(embeddings.shape[0]),
            "dtype": str(embeddings.dtype),
            **({"source": source} if source is not None else {}),
            **(
                {"idx_to_label": {int(k): v for k, v in self.idx_to_label.items()}}
                if self.idx_to_label
                else {}
            ),
            **(extra_metadata or {}),
        }
        _atomic("metadata.json", lambda p: p.write_text(json.dumps(metadata)))

        self.embedding_dir = path
        self.metadata = metadata
        return path


# ---------------------------------------------------------------------------
# Store resolution and projection-head compatibility
# ---------------------------------------------------------------------------


def resolve_store(
    *,
    embedding_dir: str | Path | None,
    require_labels: bool = False,
) -> "EmbeddingStore":
    """Load a populated embedding store from ``embedding_dir``.

    Missing, empty, and partially populated directories raise rather than
    falling back to in-memory AA encoding.
    """
    if embedding_dir is None or not str(embedding_dir).strip():
        raise ValueError(
            "embedding_dir is required. Pass a store built by "
            "contrasted-build-concat-store (2048-d AA∥3Di) or "
            "contrasted-embed (1024-d AA)."
        )

    path = Path(embedding_dir)
    if _is_populated_store(path):
        return EmbeddingStore.from_dir(path, require_labels=require_labels)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"{path} exists but is not a valid embedding directory. "
            "Remove it or point embedding_dir elsewhere."
        )
    raise FileNotFoundError(
        f"Embedding store not found at {path}. Build one with "
        "contrasted-build-concat-store (2048-d AA∥3Di) or contrasted-embed "
        "(1024-d AA)."
    )


def validate_store_for_projection_head(
    store: EmbeddingStore,
    *,
    input_dim: int,
) -> None:
    """Reject store metadata incompatible with a projection head."""
    if input_dim != 2048:
        return
    modality = store.metadata.get("modality")
    if modality != "aa_3di_concat":
        raise ValueError(
            f"Projection head input_dim 2048 requires an aa_3di_concat store, "
            f"but metadata modality is {modality!r}. Build the store with "
            "contrasted-build-concat-store."
        )


def _is_populated_store(path: Path) -> bool:
    required = ("embeddings.npy", "ids.txt", "metadata.json")
    return path.is_dir() and all((path / name).exists() for name in required)


