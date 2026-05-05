"""Data loading for protein embeddings from embedding directories."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------


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

    token = header[1:].split()[0]
    parts = token.split("|")
    if len(parts) >= 3:
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
    sequences: dict[str, str] = {}
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
                if current_id in sequences:
                    logger.warning(f"Duplicate ID '{current_id}'; keeping first")
                    current_id = None
                else:
                    sequences[current_id] = ""
            elif current_id is not None:
                sequences[current_id] += line.replace("-", "")
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


def _load_id_to_idx(embedding_dir: Path, ids: list[str]) -> dict[str, int]:
    """Load ``id_to_row.npy`` if present, else build from ``ids``."""
    mapping_path = embedding_dir / "id_to_row.npy"
    if mapping_path.exists():
        obj = np.load(mapping_path, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.shape == ():
            obj = obj.item()
        if isinstance(obj, dict):
            return {str(k): int(v) for k, v in obj.items()}
        logger.warning(
            "id_to_row.npy in %s is not a dict (got %s); rebuilding",
            embedding_dir,
            type(obj).__name__,
        )
    return {id_: i for i, id_ in enumerate(ids)}


def _metadata_value(metadata: dict, keys: list[str]):
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _load_idx_to_label(embedding_dir: Path, metadata: dict) -> dict[int, str] | None:
    idx_to_label = metadata.get("idx_to_label")
    if isinstance(idx_to_label, dict):
        return {int(k): v for k, v in idx_to_label.items()}

    sidecar = metadata.get("idx_to_label_file")
    if sidecar:
        path = embedding_dir / sidecar
        if path.exists():
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
    - ``id_to_row.npy`` (optional): precomputed ID -> row mapping
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
            if not required.exists():
                raise FileNotFoundError(f"Missing {name} in {path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        embeddings = np.load(embeddings_path, mmap_mode="r")

        dim = _metadata_value(metadata, ["dims", "dim", "embedding_dim"])
        count = _metadata_value(metadata, ["count", "num_embeddings", "n_embeddings"])
        dtype_str = _metadata_value(metadata, ["dtype"])

        if dtype_str is not None:
            expected_dtype = np.dtype(dtype_str)
            if embeddings.dtype != expected_dtype:
                raise ValueError(
                    f"embeddings.npy dtype {embeddings.dtype} != "
                    f"metadata dtype {expected_dtype}"
                )
        if dim is not None and embeddings.shape[1] != int(dim):
            raise ValueError(
                f"embeddings.npy dim {embeddings.shape[1]} != metadata dim {dim}"
            )
        if count is not None and embeddings.shape[0] != int(count):
            raise ValueError(
                f"embeddings.npy count {embeddings.shape[0]} != metadata count {count}"
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
            id_to_idx=_load_id_to_idx(path, ids),
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
        found_idx: list[int] = []
        found_ids: list[str] = []
        missing_ids: list[str] = []
        for domain_id in domain_ids:
            row = self.id_to_idx.get(domain_id)
            if row is None:
                missing_ids.append(domain_id)
            else:
                found_idx.append(row)
                found_ids.append(domain_id)
        return found_idx, found_ids, missing_ids

    def get_batch(
        self, domain_ids: list[str]
    ) -> tuple[torch.Tensor, list[str], list[str]]:
        """Fetch embeddings for ``domain_ids`` as a ``(M, D)`` float tensor."""
        found_idx, found_ids, missing_ids = self.resolve(domain_ids)
        if found_idx:
            batch_np = np.array(self.embeddings[found_idx], copy=True)
            batch = torch.from_numpy(batch_np).float()
        else:
            batch = torch.empty(0, self.dim)
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

        canonical = [
            "embeddings.npy",
            "labels.npy",
            "ids.txt",
            "id_to_row.npy",
            "metadata.json",
        ]
        existing = [n for n in canonical if (path / n).exists()]
        if existing:
            raise FileExistsError(
                f"{path} already contains {existing}; refusing to overwrite"
            )

        def _atomic(name: str, writer):
            tmp = path / f"{name}.tmp"
            writer(tmp)
            tmp.replace(path / name)

        def _save_npy(path: Path, value, *, allow_pickle: bool = False) -> None:
            with path.open("wb") as f:
                np.save(f, value, allow_pickle=allow_pickle)

        embeddings = np.ascontiguousarray(self.embeddings)
        _atomic("embeddings.npy", lambda p: _save_npy(p, embeddings))

        if self.labels is not None:
            _atomic("labels.npy", lambda p: _save_npy(p, np.asarray(self.labels)))

        _atomic(
            "ids.txt",
            lambda p: p.write_text("\n".join(self.ids) + "\n"),
        )
        _atomic(
            "id_to_row.npy",
            lambda p: _save_npy(p, dict(self.id_to_idx), allow_pickle=True),
        )

        metadata: dict = {
            "dims": int(embeddings.shape[1]),
            "count": int(embeddings.shape[0]),
            "dtype": str(embeddings.dtype),
        }
        if source is not None:
            metadata["source"] = source
        if self.idx_to_label:
            metadata["idx_to_label"] = {int(k): v for k, v in self.idx_to_label.items()}
        if extra_metadata:
            metadata.update(extra_metadata)
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
    """Return an :class:`EmbeddingStore` according to this precedence:

    1. ``embedding_dir`` set and populated -> load from disk.
    2. ``embedding_dir`` set but missing/empty -> encode ``fasta_paths`` and,
       if possible, persist the 4-file layout to ``embedding_dir``.
    3. ``embedding_dir`` unset -> encode ``fasta_paths`` in-memory.

    A partially populated directory raises :class:`FileExistsError` rather
    than overwriting. The ``contrasted.embed`` module is imported lazily so
    users who already have a prepared directory pay no encoding cost.
    """
    if embedding_dir is not None:
        path = Path(embedding_dir)
        if _is_populated_store(path):
            return EmbeddingStore.from_dir(path, require_labels=require_labels)
        if path.exists() and any(path.iterdir()) and not _is_populated_store(path):
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
    required = ["embeddings.npy", "ids.txt", "metadata.json"]
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:  # type: ignore[override]
        data_idx = self.indices[idx]
        embedding = self.embeddings[data_idx]
        if isinstance(embedding, np.ndarray):
            # Copy to defeat memory-mapped read-only arrays, which torch
            # accepts but warns about.
            embedding = torch.from_numpy(np.array(embedding, copy=True)).float()
        label = self.labels[data_idx]
        if isinstance(label, np.ndarray):
            label = int(label)
        else:
            label = int(label.item())
        return embedding, label


class EmbeddingDataModule(L.LightningDataModule):
    """Lightning data module backed by an :class:`EmbeddingStore`.

    Each split (train/val/test) is defined by a FASTA file whose headers
    name domains present in ``embedding_dir``. ``test_fasta`` may be a
    single file, a directory of FASTAs, or a list of paths; each becomes
    one entry in :attr:`test_datasets`.
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
        if self.store is None:
            return 0
        if self.store.idx_to_label:
            return len(self.store.idx_to_label)
        return 0

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
            unique = np.unique(self.store.labels).tolist()
            self.store.idx_to_label = {int(i): str(int(i)) for i in unique}
        logger.info(
            f"Loaded {len(self.store.ids)} embeddings, "
            f"{self.store.dim} dims, {self.num_classes} classes"
        )

    def _create_dataset(self, fasta_path: Path) -> EmbeddingDataset:
        assert self.store is not None and self.store.labels is not None

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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        loaders = [self._dataloader(ds) for ds in self.test_datasets.values()]
        if len(loaders) == 1:
            return loaders[0]
        return loaders

    def _dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
