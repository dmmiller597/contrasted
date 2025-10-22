import h5py
import lightning as L
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple

from contrasted.utils import extract_domain_id, load_h5_keys_from_fasta, load_labels


class CathEmbeddingDataset(Dataset):
    """CATH protein embeddings from HDF5."""

    def __init__(
        self,
        h5_path: Path,
        h5_keys: List[str],
        labels: Dict[str, int],
        cache_embeddings: bool = True,
    ):
        self.h5_path = h5_path
        self.cache_embeddings = cache_embeddings
        self._h5_file = None
        self._embedding_cache: Optional[Dict[str, torch.Tensor]] = None
        self.samples = self._filter_samples(h5_keys, labels)
        
        # Pre-load all embeddings if caching is enabled
        if self.cache_embeddings:
            self._load_all_embeddings()

    def _filter_samples(self, h5_keys: List[str], labels: Dict[str, int]) -> List[Tuple[str, int]]:
        """Keep only keys with valid labels."""
        samples = []
        for key in h5_keys:
            try:
                domain_id = extract_domain_id(key)
                if domain_id in labels:
                    samples.append((key, labels[domain_id]))
            except (ValueError, IndexError):
                continue
        return samples

    def _load_all_embeddings(self) -> None:
        """Pre-load all embeddings into memory for fast access."""
        self._embedding_cache = {}
        with h5py.File(self.h5_path, 'r') as f:
            for key, _ in self.samples:
                self._embedding_cache[key] = torch.from_numpy(f[key][:]).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        key, label = self.samples[idx]
        
        # Use cached embedding if available
        if self._embedding_cache is not None:
            embedding = self._embedding_cache[key]
        else:
            # Lazy load from disk
            if self._h5_file is None:
                self._h5_file = h5py.File(self.h5_path, 'r')
            embedding = torch.from_numpy(self._h5_file[key][:]).float()
        
        return embedding, label

    def __del__(self):
        """Ensure HDF5 file is closed."""
        if self._h5_file is not None:
            self._h5_file.close()


class CathDataModule(L.LightningDataModule):
    """CATH protein superfamily classification data."""

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
        cache_embeddings: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_fasta = Path(train_fasta)
        self.val_fasta = Path(val_fasta)
        self.test_fasta = Path(test_fasta)
        self.label_file = Path(label_file)
        self.embedding_file = Path(embedding_file)
        self.pin_memory = pin_memory and not torch.backends.mps.is_available()
        self.cache_embeddings = cache_embeddings
        
        self.labels: Optional[Dict[str, int]] = None
        self.idx_to_label: Optional[Dict[int, str]] = None

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_label) if self.idx_to_label else 0

    def setup(self, stage: Optional[str] = None):
        if self.labels is None:
            self.labels, self.idx_to_label = load_labels(self.label_file)

        if stage in ("fit", None):
            self.train_dataset = self._create_dataset(self.train_fasta)
            self.val_dataset = self._create_dataset(self.val_fasta)
        
        if stage in ("test", None):
            self.test_dataset = self._create_dataset(self.test_fasta)

    def _create_dataset(self, fasta_path: Path) -> CathEmbeddingDataset:
        return CathEmbeddingDataset(
            self.embedding_file,
            load_h5_keys_from_fasta(fasta_path),
            self.labels,
            cache_embeddings=self.cache_embeddings
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )
