import h5py
import torch
import lightning as L
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def parse_fasta_header(header: str) -> str:
    """Extract domain_id from FASTA header.
    
    Format: >cath|{cath_release}|{domain_id}/{start}-{end}
    Returns: domain_id (e.g., '12e8H01')
    """
    parts = header.strip().lstrip('>').split('|')
    if len(parts) >= 3:
        return parts[2].split('/')[0]
    raise ValueError(f"Invalid FASTA header: {header}")


def fasta_to_h5_key(header: str) -> str:
    """Convert FASTA header to HDF5 key format.
    
    FASTA: >cath|4_4_0|12e8H01/1-113
    H5:    cath|4_4_0|12e8H01_1-113
    """
    return header.strip().lstrip('>').replace('/', '_')


def load_h5_keys_from_fasta(fasta_path: Path) -> List[str]:
    """Read FASTA file and return list of HDF5 keys."""
    h5_keys = []
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                try:
                    h5_keys.append(fasta_to_h5_key(line))
                except ValueError as e:
                    logger.warning(f"Could not parse header: {line.strip()} - {e}")
    return h5_keys


def load_labels(label_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load CATH superfamily labels.
    
    File format: {domain_id}\\t{superfamily_code}
    Returns: (domain_id -> sf_idx, sf_idx -> sf_code)
    """
    id_to_sf_idx: Dict[str, int] = {}
    sf_to_idx: Dict[str, int] = {}
    
    with open(label_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                domain_id, superfamily = parts[0], parts[1]
                if superfamily not in sf_to_idx:
                    sf_to_idx[superfamily] = len(sf_to_idx)
                id_to_sf_idx[domain_id] = sf_to_idx[superfamily]

    idx_to_sf = {v: k for k, v in sf_to_idx.items()}
    return id_to_sf_idx, idx_to_sf


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
                domain_id = h5_key.split('|')[2].split('_')[0]
                if domain_id in labels:
                    self.h5_keys.append(h5_key)
                    self.domain_ids.append(domain_id)
                else:
                    skipped += 1
            except (ValueError, IndexError):
                skipped += 1
        
        if skipped > 0:
            logger.info(f"Skipped {skipped} samples without labels. Kept {len(self.h5_keys)}.")

    @property
    def h5_file(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
        return self._h5_file

    def __len__(self) -> int:
        return len(self.h5_keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        h5_key = self.h5_keys[idx]
        domain_id = self.domain_ids[idx]
        
        try:
            embedding = self.h5_file[h5_key][:]
            label = self.labels[domain_id]
            return torch.from_numpy(embedding).float(), label
        except KeyError:
            logger.error(f"Missing embedding for key: {h5_key}")
            raise KeyError(f"Missing embedding for key: {h5_key}")


def worker_init_fn(worker_id: int):
    """Initialize worker with separate HDF5 file handle."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker_info.dataset._h5_file = None


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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.num_workers > 0,
        )
