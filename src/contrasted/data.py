"""Data loading for protein embeddings from embedding directories."""

import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def parse_fasta_header(header: str) -> str:
    """Extract domain_id from FASTA header.

    Supported formats:
    - CATH: >cath|{cath_release}|{domain_id}/{start}-{end}
    - TED/AlphaFold: >AF-..._TED03
    - Fallback: take first token after ">" (before whitespace)
    """
    header = header.strip()
    if not header.startswith(">"):
        raise ValueError(f"Invalid FASTA header: {header}")

    token = header[1:].split()[0]
    parts = token.split("|")
    if len(parts) >= 3:
        return parts[2].split("/")[0]
    return token.split("/", 1)[0]


def load_domain_ids_from_fasta(fasta_path: str | Path) -> list[str]:
    """Read FASTA file and return list of domain IDs."""
    domain_ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                try:
                    domain_ids.append(parse_fasta_header(line))
                except ValueError as e:
                    logger.warning(f"Could not parse header: {line.strip()} - {e}")
    return domain_ids


def resolve_fasta_paths(fasta_input: str | Path) -> dict[str, Path]:
    """Resolve FASTA paths - handles single file or directory."""
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


def read_ids_txt(path: Path) -> list[str]:
    """Read ids.txt with one ID per line."""
    ids = []
    with open(path) as f:
        for line in f:
            id_ = line.strip()
            if id_:
                ids.append(id_)
    return ids


def load_id_to_idx(embedding_dir: str | Path, ids: list[str]) -> dict[str, int]:
    """Load or build ID to index mapping for an embedding directory.

    If id_to_row.npy exists, loads the mapping from there.
    Otherwise, builds mapping from the ids list.
    """
    embedding_dir = Path(embedding_dir)
    mapping_path = embedding_dir / "id_to_row.npy"
    if mapping_path.exists():
        mapping_obj = np.load(mapping_path, allow_pickle=True)
        if isinstance(mapping_obj, np.ndarray) and mapping_obj.shape == ():
            mapping_obj = mapping_obj.item()
        if isinstance(mapping_obj, dict):
            return {str(k): int(v) for k, v in mapping_obj.items()}
    return {id_: i for i, id_ in enumerate(ids)}


def read_labels_npy(path: Path) -> np.ndarray:
    """Read labels.npy as int64."""
    labels = np.load(path)
    return np.asarray(labels, dtype=np.int64)


def _load_metadata(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_metadata_value(metadata: dict, keys: list[str]) -> int | str | None:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def load_embedding_dir(
    embedding_dir: str | Path,
    *,
    require_labels: bool = False,
) -> tuple[np.ndarray, list[str], np.ndarray | None, dict[int, str] | None]:
    """Load embedding directory containing embeddings.npy, ids.txt, metadata.json."""
    embedding_dir = Path(embedding_dir)
    embeddings_path = embedding_dir / "embeddings.npy"
    ids_path = embedding_dir / "ids.txt"
    metadata_path = embedding_dir / "metadata.json"
    labels_path = embedding_dir / "labels.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings.npy in {embedding_dir}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing ids.txt in {embedding_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {embedding_dir}")

    metadata = _load_metadata(metadata_path)
    dim = _get_metadata_value(metadata, ["dims", "dim", "embedding_dim"])
    count = _get_metadata_value(metadata, ["count", "num_embeddings", "n_embeddings"])
    dtype_str = _get_metadata_value(metadata, ["dtype"])

    embeddings = np.load(embeddings_path, mmap_mode="r")
    if dtype_str is not None:
        expected_dtype = np.dtype(dtype_str)
        if embeddings.dtype != expected_dtype:
            raise ValueError(
                "embeddings.npy dtype "
                f"{embeddings.dtype} != metadata dtype {expected_dtype}"
            )
    if dim is not None and embeddings.shape[1] != int(dim):
        raise ValueError(
            f"embeddings.npy dim {embeddings.shape[1]} != metadata dim {dim}"
        )
    if count is not None and embeddings.shape[0] != int(count):
        raise ValueError(
            f"embeddings.npy count {embeddings.shape[0]} != metadata count {count}"
        )

    ids = read_ids_txt(ids_path)
    if len(ids) != embeddings.shape[0]:
        raise ValueError(
            "ids.txt has "
            f"{len(ids)} entries but embeddings.npy has {embeddings.shape[0]}"
        )

    labels = None
    if labels_path.exists():
        labels = read_labels_npy(labels_path)
        if labels.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "labels.npy has "
                f"{labels.shape[0]} entries but embeddings.npy has "
                f"{embeddings.shape[0]}"
            )
    elif require_labels:
        raise FileNotFoundError(f"Missing labels.npy in {embedding_dir}")

    idx_to_label = metadata.get("idx_to_label")
    if isinstance(idx_to_label, dict):
        idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    else:
        idx_to_label = None

    return embeddings, ids, labels, idx_to_label


class EmbeddingDataset(Dataset[tuple[torch.Tensor, int]]):
    """Protein embeddings from embedding directory.

    The embedding directory contains:
        - embeddings.npy: (N, D) float16/float32
        - labels.npy: (N,) int64
        - ids.txt: one ID per line

    This dataset uses a subset defined by indices (from fasta file filtering).
    """

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
            # Copy needed for memory-mapped arrays (mmap_mode="r" returns read-only)
            embedding = torch.from_numpy(np.array(embedding)).float()
        label = self.labels[data_idx]
        if isinstance(label, np.ndarray):
            label = int(label)
        else:
            label = int(label.item())
        return embedding, label


class EmbeddingDataModule(L.LightningDataModule):
    """Protein domain classification data using embedding directories."""

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

        self.embeddings: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.id_to_idx: dict[str, int] | None = None
        self.idx_to_label: dict[int, str] | None = None
        self.test_datasets: dict[str, EmbeddingDataset] | None = None

    def _resolve_test_paths(self, test_fasta: str | list[str]) -> dict[str, Path]:
        if isinstance(test_fasta, list):
            return {Path(p).stem: Path(p) for p in test_fasta}
        return resolve_fasta_paths(Path(test_fasta))

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_label) if self.idx_to_label else 0

    def setup(self, stage: str | None = None):
        if self.embeddings is None:
            self._load_embedding_dir()

        if stage in ("fit", None):
            self.train_dataset = self._create_dataset(self.train_fasta)
            self.val_dataset = self._create_dataset(self.val_fasta)

        if stage in ("test", None):
            self.test_datasets = {
                name: self._create_dataset(path)
                for name, path in self.test_fasta_paths.items()
            }
            if self.test_datasets:
                self.test_dataset = next(iter(self.test_datasets.values()))

    def _load_embedding_dir(self):
        """Load embeddings, labels, and IDs from embedding directory."""
        logger.info(f"Loading embeddings from: {self.embedding_dir}")

        embeddings, ids, labels, idx_to_label = load_embedding_dir(
            self.embedding_dir,
            require_labels=True,
        )
        assert labels is not None

        self.embeddings = embeddings
        self.labels = labels
        self.id_to_idx = {id_: i for i, id_ in enumerate(ids)}

        if idx_to_label:
            self.idx_to_label = idx_to_label
        else:
            unique_labels = np.unique(labels).tolist()
            self.idx_to_label = {int(i): str(int(i)) for i in unique_labels}

        logger.info(
            f"Loaded {len(ids)} embeddings, "
            f"{self.embeddings.shape[1]} dims, "
            f"{len(self.idx_to_label)} classes"
        )

    def _create_dataset(self, fasta_path: Path) -> EmbeddingDataset:
        """Create dataset for a split defined by fasta file."""
        assert self.embeddings is not None and self.labels is not None
        assert self.id_to_idx is not None

        domain_ids = load_domain_ids_from_fasta(fasta_path)

        indices = []
        missing = 0
        for domain_id in domain_ids:
            if domain_id in self.id_to_idx:
                indices.append(self.id_to_idx[domain_id])
            else:
                missing += 1

        if missing > 0:
            logger.warning(
                f"{fasta_path.name}: {missing}/{len(domain_ids)} domains not found"
            )

        if len(indices) == 0:
            raise ValueError(
                f"{fasta_path.name}: No valid samples found. "
                f"All {len(domain_ids)} domains are missing from embedding directory."
            )

        logger.info(f"{fasta_path.name}: {len(indices)} samples")

        return EmbeddingDataset(
            self.embeddings,
            self.labels,
            indices,
        )

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
        assert self.test_datasets is not None
        if len(self.test_datasets) == 1:
            return self._dataloader(self.test_dataset)
        return [self._dataloader(ds) for ds in self.test_datasets.values()]

    def _dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
