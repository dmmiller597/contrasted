"""Data loading for protein embeddings using PyTorch .pt files."""

import logging
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def parse_fasta_header(header: str) -> str:
    """Extract domain_id from FASTA header.

    Format: >cath|{cath_release}|{domain_id}/{start}-{end}
    Returns: domain_id (e.g., '12e8H01')
    """
    parts = header.strip().lstrip(">").split("|")
    if len(parts) >= 3:
        return parts[2].split("/")[0]
    raise ValueError(f"Invalid FASTA header: {header}")


def load_domain_ids_from_fasta(fasta_path: Path) -> list[str]:
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


def resolve_fasta_paths(fasta_input: Path) -> dict[str, Path]:
    """Resolve FASTA paths - handles single file or directory."""
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


class EmbeddingDataset(Dataset[tuple[torch.Tensor, int]]):
    """Protein embeddings from .pt file.

    The .pt file contains:
        - embeddings: Tensor[N, D] - all embeddings (float16 or float32)
        - labels: Tensor[N] - superfamily indices
        - ids: List[str] - domain IDs

    This dataset uses a subset defined by indices (from fasta file filtering).
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        indices: list[int],
    ):
        self.embeddings = embeddings
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:  # type: ignore[override]
        data_idx = self.indices[idx]
        return self.embeddings[data_idx], self.labels[data_idx].item()  # type: ignore[return-value]


class EmbeddingDataModule(L.LightningDataModule):
    """Protein domain classification data using .pt files."""

    def __init__(
        self,
        train_fasta: str,
        val_fasta: str,
        test_fasta: str | list[str],
        embedding_file: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        label_file: str | None = None,  # kept for config compatibility
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_fasta = Path(train_fasta)
        self.val_fasta = Path(val_fasta)
        self.test_fasta_paths = self._resolve_test_paths(test_fasta)
        self.embedding_file = Path(embedding_file)
        self.pin_memory = pin_memory and not torch.backends.mps.is_available()

        self.embeddings: torch.Tensor | None = None
        self.labels: torch.Tensor | None = None
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
            self._load_pt_file()

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

    def _load_pt_file(self):
        """Load embeddings, labels, and IDs from .pt file."""
        logger.info(f"Loading embeddings from: {self.embedding_file}")

        try:
            data = torch.load(
                self.embedding_file,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            logger.info("Loaded with memory mapping")
        except TypeError:
            data = torch.load(
                self.embedding_file, map_location="cpu", weights_only=False
            )
            logger.info("Loaded without memory mapping (PyTorch < 2.1)")

        self.embeddings = data["embeddings"].float()
        self.labels = data["labels"]

        ids = data["ids"]
        self.id_to_idx = {id_: i for i, id_ in enumerate(ids)}

        if "idx_to_label" in data:
            self.idx_to_label = data["idx_to_label"]
        else:
            unique_labels = self.labels.unique().tolist()
            self.idx_to_label = {i: str(i) for i in unique_labels}

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

    def get_test_dataloader(self, name: str) -> DataLoader | None:
        if self.test_datasets and name in self.test_datasets:
            return self._dataloader(self.test_datasets[name])
        return None

    def get_test_names(self) -> list[str]:
        return list(self.test_fasta_paths.keys())

    def _dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
