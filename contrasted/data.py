import h5py
import lightning as L
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Union

from contrasted.utils import extract_domain_id, load_h5_keys_from_fasta, load_labels, resolve_fasta_paths



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
        self._embedding_cache: Optional[Dict[str, torch.Tensor]] = None
        self.samples = [
            (key, labels[domain_id])
            for key in h5_keys
            if (domain_id := extract_domain_id(key)) in labels
        ]
        
        if cache_embeddings:
            self._embedding_cache = {}
            with h5py.File(self.h5_path, 'r') as f:
                for key, _ in self.samples:
                    self._embedding_cache[key] = torch.from_numpy(f[key][:]).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        key, label = self.samples[idx]
        
        if self._embedding_cache is not None:
            return self._embedding_cache[key], label
        
        with h5py.File(self.h5_path, 'r') as f:
            return torch.from_numpy(f[key][:]).float(), label


class CathDataModule(L.LightningDataModule):
    """CATH protein superfamily classification data."""
    
    def __init__(
        self,
        train_fasta: str,
        val_fasta: str,
        test_fasta: Union[str, List[str]],
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
        self.test_fasta_paths = self._resolve_test_paths(test_fasta)
        self.label_file = Path(label_file)
        self.embedding_file = Path(embedding_file)
        self.pin_memory = pin_memory and not torch.backends.mps.is_available()
        self.cache_embeddings = cache_embeddings
        
        self.labels: Optional[Dict[str, int]] = None
        self.idx_to_label: Optional[Dict[int, str]] = None
        self.test_datasets: Optional[Dict[str, CathEmbeddingDataset]] = None

    def _resolve_test_paths(self, test_fasta: Union[str, List[str]]) -> Dict[str, Path]:
        if isinstance(test_fasta, list):
            return {Path(p).stem: Path(p) for p in test_fasta}
        return resolve_fasta_paths(Path(test_fasta))

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
            self.test_datasets = {
                name: self._create_dataset(path)
                for name, path in self.test_fasta_paths.items()
            }
            if self.test_datasets:
                self.test_dataset = next(iter(self.test_datasets.values()))

    def _create_dataset(self, fasta_path: Path) -> CathEmbeddingDataset:
        return CathEmbeddingDataset(
            self.embedding_file,
            load_h5_keys_from_fasta(fasta_path),
            self.labels,
            self.cache_embeddings
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

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if len(self.test_datasets) == 1:
            return self._dataloader(self.test_dataset)
        return [self._dataloader(ds) for ds in self.test_datasets.values()]
    
    def get_test_dataloader(self, name: str) -> Optional[DataLoader]:
        if self.test_datasets and name in self.test_datasets:
            return self._dataloader(self.test_datasets[name])
        return None
    
    def get_test_names(self) -> List[str]:
        return list(self.test_fasta_paths.keys())

    def _dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )
