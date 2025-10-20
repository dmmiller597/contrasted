import h5py
import torch
import lightning as L
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Optional, Tuple
from collections import Counter
import numpy as np
import logging

from contrasted.utils import load_h5_keys_from_fasta, load_labels, extract_domain_id

logger = logging.getLogger(__name__)


class CathEmbeddingDataset(Dataset):
    """PyTorch Dataset for CATH embeddings stored in HDF5."""

    def __init__(self, h5_path: Path, h5_keys: List[str], labels: Dict[str, int]):
        self.h5_path = h5_path
        self.labels = labels
        self._h5_file = None
        
        self.h5_keys = []
        self.domain_ids = []
        skipped = 0
        
        for h5_key in h5_keys:
            try:
                domain_id = extract_domain_id(h5_key)
                if domain_id in labels:
                    self.h5_keys.append(h5_key)
                    self.domain_ids.append(domain_id)
                else:
                    skipped += 1
            except (ValueError, IndexError):
                skipped += 1
        
        if skipped > 0:
            logger.info(f"Skipped {skipped} samples without labels. Kept {len(self.h5_keys)}.")

    def __len__(self) -> int:
        return len(self.h5_keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        h5_key = self.h5_keys[idx]
        domain_id = self.domain_ids[idx]
        
        # Lazy-open file handle per worker (safe across process boundaries)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        
        try:
            embedding = self._h5_file[h5_key][:]
            label = self.labels[domain_id]
            return torch.from_numpy(embedding).float(), label
        except KeyError:
            logger.error(f"Missing embedding for key: {h5_key}")
            raise KeyError(f"Missing embedding for key: {h5_key}")


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
        self.pin_memory = pin_memory
        self.use_weighted_sampling = use_weighted_sampling

        if self.pin_memory and torch.backends.mps.is_available():
            logger.warning("pin_memory=True is not supported on MPS. Disabling pin_memory.")
            self.pin_memory = False

        self.labels: Optional[Dict[str, int]] = None
        self.idx_to_label: Optional[Dict[int, str]] = None

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_label) if self.idx_to_label else 0
        
    def prepare_data(self):
        for path in [self.train_fasta, self.val_fasta, self.test_fasta, 
                     self.label_file, self.embedding_file]:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

    def setup(self, stage: Optional[str] = None):
        self.labels, self.idx_to_label = load_labels(self.label_file)
        logger.info(f"Loaded {len(self.idx_to_label)} superfamilies, "
                   f"{len(self.labels)} domain IDs")

        if stage == "fit" or stage is None:
            train_keys = load_h5_keys_from_fasta(self.train_fasta)
            val_keys = load_h5_keys_from_fasta(self.val_fasta)
            
            self.train_dataset = CathEmbeddingDataset(
                self.embedding_file, train_keys, self.labels
            )
            self.val_dataset = CathEmbeddingDataset(
                self.embedding_file, val_keys, self.labels
            )
            
            logger.info(f"Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            test_keys = load_h5_keys_from_fasta(self.test_fasta)
            self.test_dataset = CathEmbeddingDataset(
                self.embedding_file, test_keys, self.labels
            )
            logger.info(f"Test: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        if self.use_weighted_sampling:
            # Compute sample weights: inverse frequency
            labels = [self.labels[domain_id] for domain_id in self.train_dataset.domain_ids]
            class_counts = Counter(labels)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[label] for label in labels]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
