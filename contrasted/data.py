import h5py
import torch
import lightning as L
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

from contrasted.utils import load_h5_keys_from_fasta, load_labels, extract_domain_id

logger = logging.getLogger(__name__)


class CathEmbeddingDataset(Dataset):
    """Dataset for CATH embeddings stored in HDF5."""

    def __init__(self, h5_path: Path, h5_keys: List[str], labels: Dict[str, int]):
        self.h5_path = h5_path
        self._h5_file = None
        
        # Filter: only keep keys with valid labels
        self.samples = []  # (h5_key, label) tuples
        for h5_key in h5_keys:
            try:
                domain_id = extract_domain_id(h5_key)
                if domain_id in labels:
                    self.samples.append((h5_key, labels[domain_id]))
            except (ValueError, IndexError):
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        h5_key, label = self.samples[idx]
        
        # Lazy-open H5 file per worker (multiprocessing-safe)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        
        embedding = self._h5_file[h5_key][:]
        return torch.from_numpy(embedding).float(), label


class CathDataModule(L.LightningDataModule):
    """Lightning DataModule for CATH protein superfamily classification."""

    def __init__(
        self,
        train_fasta: str,
        val_fasta: str,
        test_fasta: str,
        label_file: str,
        embedding_file: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_weighted_sampling: bool = False,
    ):
        super().__init__()
        self.train_fasta = Path(train_fasta)
        self.val_fasta = Path(val_fasta)
        self.test_fasta = Path(test_fasta)
        self.label_file = Path(label_file)
        self.embedding_file = Path(embedding_file)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_weighted_sampling = use_weighted_sampling
        
        # Disable pin_memory on MPS (not supported)
        self.pin_memory = pin_memory and not torch.backends.mps.is_available()

        self.labels: Optional[Dict[str, int]] = None
        self.idx_to_label: Optional[Dict[int, str]] = None

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_label) if self.idx_to_label else 0

    def setup(self, stage: Optional[str] = None):
        # Load labels once
        self.labels, self.idx_to_label = load_labels(self.label_file)

        if stage == "fit" or stage is None:
            self.train_dataset = CathEmbeddingDataset(
                self.embedding_file,
                load_h5_keys_from_fasta(self.train_fasta),
                self.labels
            )
            self.val_dataset = CathEmbeddingDataset(
                self.embedding_file,
                load_h5_keys_from_fasta(self.val_fasta),
                self.labels
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = CathEmbeddingDataset(
                self.embedding_file,
                load_h5_keys_from_fasta(self.test_fasta),
                self.labels
            )

    def _get_sampler(self):
        """Create weighted sampler for class balancing."""
        labels = [label for _, label in self.train_dataset.samples]
        class_counts = Counter(labels)
        weights = [1.0 / class_counts[label] for label in labels]
        return WeightedRandomSampler(weights, len(weights), replacement=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self._get_sampler() if self.use_weighted_sampling else None,
            shuffle=not self.use_weighted_sampling,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset)

    def _create_dataloader(self, dataset) -> DataLoader:
        """Create standard dataloader for val/test."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
