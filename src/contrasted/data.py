"""Data loading for protein embeddings from embedding directories."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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


def _read_ids_txt(path: Path) -> list[str]:
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

        ids = _read_ids_txt(ids_path)
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
        return path


# ---------------------------------------------------------------------------
# Store resolution: cached dir vs. on-the-fly encoding from FASTA
# ---------------------------------------------------------------------------


def resolve_store(
    *,
    embedding_dir: str | Path | None,
    fasta_paths: list[Path] | None = None,
    labels_path: str | Path | None = None,
    encode_config=None,
    require_labels: bool = False,
) -> "EmbeddingStore":
    """Return an ``EmbeddingStore`` according to this precedence:

    1. ``embedding_dir`` set and populated -> load from disk.
    2. ``embedding_dir`` set but missing/empty -> encode ``fasta_paths`` and,
       if possible, persist the 4-file layout to ``embedding_dir``.
    3. ``embedding_dir`` unset -> encode ``fasta_paths`` in-memory.

    A partially populated directory raises ``FileExistsError`` rather
    than overwriting. The ``contrasted.embed`` module is imported lazily so
    users who already have a prepared directory pay no encoding cost.
    """
    if embedding_dir is not None:
        path = Path(embedding_dir)
        if _is_populated_store(path):
            return EmbeddingStore.from_dir(path, require_labels=require_labels)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(
                f"{path} exists but is not a valid embedding directory. "
                "Remove it or point embedding_dir elsewhere."
            )

    if not fasta_paths:
        raise ValueError(
            "No embedding_dir and no fasta_paths provided; cannot build a store."
        )

    from contrasted.embed import encode_fasta  # lazy import

    store = encode_fasta(
        fasta_paths,
        labels_path=labels_path,
        config=encode_config,
    )
    if require_labels and store.labels is None:
        raise FileNotFoundError(
            "require_labels=True but no labels_path was supplied for encoding."
        )

    if embedding_dir is not None:
        store.save(embedding_dir, source=",".join(str(p) for p in fasta_paths))
        logger.info(f"Cached on-the-fly embeddings to: {embedding_dir}")

    return store


def _is_populated_store(path: Path) -> bool:
    required = ("embeddings.npy", "ids.txt", "metadata.json")
    return path.is_dir() and all((path / name).exists() for name in required)


# ---------------------------------------------------------------------------
# Dataset / DataModule
# ---------------------------------------------------------------------------


class EmbeddingDataset(Dataset[tuple[torch.Tensor, int]]):
    """A subset of an embedding array, defined by row indices."""

    def __init__(
        self,
        embeddings: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        indices: list[int],
    ):
        self.embeddings = embeddings
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        data_idx = self.indices[index]
        embedding = self.embeddings[data_idx]
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(np.array(embedding, copy=True)).float()
        return embedding, int(self.labels[data_idx])


class EmbeddingDataModule(L.LightningDataModule):
    """Lightning data module backed by an ``EmbeddingStore``.

    Each split (train/val/test) is defined by a FASTA file whose headers
    name domains present in ``embedding_dir``. ``test_fasta`` may be a
    single file, a directory of FASTAs, or a list of paths; each becomes
    one entry in ``test_datasets``.
    """

    def __init__(
        self,
        train_fasta: str,
        val_fasta: str,
        test_fasta: str | list[str],
        embedding_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_fasta = Path(train_fasta)
        self.val_fasta = Path(val_fasta)
        self.test_fasta_paths = self._resolve_test_paths(test_fasta)
        self.embedding_dir = Path(embedding_dir)
        self.pin_memory = pin_memory and not torch.backends.mps.is_available()

        self.store: EmbeddingStore | None = None
        self.test_datasets: dict[str, EmbeddingDataset] = {}

    @staticmethod
    def _resolve_test_paths(
        test_fasta: str | list[str],
    ) -> dict[str, Path]:
        if isinstance(test_fasta, list):
            return {Path(p).stem: Path(p) for p in test_fasta}
        return resolve_fasta_paths(Path(test_fasta))

    @property
    def num_classes(self) -> int:
        return self.store.num_classes if self.store is not None else 0

    def setup(self, stage: str | None = None):
        if self.store is None:
            self._load_store()

        if stage in ("fit", None):
            self.train_dataset = self._create_dataset(self.train_fasta)
            self.val_dataset = self._create_dataset(self.val_fasta)

        if stage in ("test", None):
            self.test_datasets = {
                name: self._create_dataset(path)
                for name, path in self.test_fasta_paths.items()
            }

    def _load_store(self) -> None:
        logger.info(f"Loading embeddings from: {self.embedding_dir}")
        self.store = EmbeddingStore.from_dir(self.embedding_dir, require_labels=True)
        if self.store.idx_to_label is None and self.store.labels is not None:
            if self.store.labels.min() < 0:
                raise ValueError(
                    f"labels.npy in {self.embedding_dir} has negative values; "
                    "class indices must be non-negative."
                )
            self.store.idx_to_label = {
                i: str(i) for i in range(int(self.store.labels.max()) + 1)
            }
        logger.info(
            f"Loaded {len(self.store.ids)} embeddings, "
            f"{self.store.dim} dims, {self.num_classes} classes"
        )

    def _create_dataset(self, fasta_path: Path) -> EmbeddingDataset:
        if self.store is None or self.store.labels is None:
            raise RuntimeError("Embedding store with labels has not been loaded.")

        domain_ids = load_domain_ids_from_fasta(fasta_path)
        indices, _, missing_ids = self.store.resolve(domain_ids)

        if missing_ids:
            logger.warning(
                f"{fasta_path.name}: {len(missing_ids)}/{len(domain_ids)} "
                "domains not found"
            )
        if not indices:
            raise ValueError(
                f"{fasta_path.name}: No valid samples found. "
                f"All {len(domain_ids)} domains are missing from "
                "embedding directory."
            )
        logger.info(f"{fasta_path.name}: {len(indices)} samples")

        return EmbeddingDataset(self.store.embeddings, self.store.labels, indices)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        loaders = [self._dataloader(ds) for ds in self.test_datasets.values()]
        return loaders[0] if len(loaders) == 1 else loaders

    def _dataloader(self, dataset: Dataset, *, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
