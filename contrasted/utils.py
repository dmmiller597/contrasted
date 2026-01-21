import random
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import h5py
import lmdb

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


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


def normalize_h5_key(h5_key: str) -> str:
    """Normalize H5 key to underscore format for consistent access.
    
    Converts pipe format to underscore format:
    - Input: cath|4_4_0|12e8H01_1-113
    - Output: cath_4_4_0_12e8H01_1-113
    
    If already in underscore format, returns as-is.
    """
    if '|' in h5_key:
        # Convert pipe format to underscore format
        return h5_key.replace('|', '_')
    return h5_key


def extract_domain_id(h5_key: str) -> str:
    """Extract domain ID from HDF5 key.
    
    Handles multiple formats including multi-segment position ranges:
    - Format 1 (pipe with slash): cath|4_4_0|2axqA02/125-248_349-387 -> 2axqA02
    - Format 2 (pipe with underscore): cath|4_4_0|12e8H01_1-113 -> 12e8H01
    - Format 3 (underscore): cath_4_4_0_107lA00_1-162 -> 107lA00
    - Format 4 (non-CATH): AF-A0A1A7ZDH5-F1-model_v4_TED01 -> returns as-is
    """
    # Try pipe format first (CATH format)
    if '|' in h5_key:
        domain_part = h5_key.split('|')[2]
        
        # Check for slash separator (newer format): domain_id/pos1-pos2_pos3-pos4
        if '/' in domain_part:
            return domain_part.split('/')[0]
        
        # Handle underscore format: domain_id_pos1-pos2 or just domain_id
        if '_' in domain_part:
            parts = domain_part.split('_')
            # Find first part that doesn't look like a position range
            domain_id_parts = []
            for part in parts:
                # Position ranges contain '-' and all non-dash chars are digits
                if '-' in part and part.replace('-', '').isdigit():
                    break  # Stop at first position range
                domain_id_parts.append(part)
            if domain_id_parts:
                return domain_id_parts[0]
            return parts[0]
        
        return domain_part
    
    # Handle underscore format: cath_4_4_0_107lA00_1-162
    # Split by underscore: ['cath', '4', '4', '0', '107lA00', '1-162']
    # Domain ID is at index 4 (5th part)
    parts = h5_key.split('_')
    if len(parts) >= 5 and parts[0] == 'cath':
        return parts[4]
    
    # For non-CATH formats (e.g., AlphaFold, Uniprot), return the full key as ID
    return h5_key


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
            if not line.startswith("#"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    domain_id, superfamily = parts[0], parts[1]
                    if superfamily not in sf_to_idx:
                        sf_to_idx[superfamily] = len(sf_to_idx)
                    id_to_sf_idx[domain_id] = sf_to_idx[superfamily]

    return id_to_sf_idx, {v: k for k, v in sf_to_idx.items()}


def resolve_fasta_paths(fasta_input: Path) -> Dict[str, Path]:
    """Resolve FASTA paths - handles single file or directory."""
    if fasta_input.is_dir():
        fasta_files = sorted(fasta_input.glob("*.fasta")) + sorted(fasta_input.glob("*.fa"))
        if not fasta_files:
            logger.warning(f"No FASTA files found in directory: {fasta_input}")
            return {}
        logger.info(f"Found {len(fasta_files)} FASTA files in {fasta_input}")
        return {f.stem: f for f in fasta_files}
    if fasta_input.is_file():
        return {fasta_input.stem: fasta_input}
    logger.warning(f"FASTA path not found: {fasta_input}")
    return {}


class EmbeddingReader:
    """Unified interface for reading embeddings from HDF5 or LMDB."""
    
    def __init__(self, embedding_path: Path):
        """Initialize reader for HDF5 or LMDB embedding storage.
        
        Args:
            embedding_path: Path to HDF5 file (.h5) or LMDB directory
        """
        self.embedding_path = Path(embedding_path)
        self.is_lmdb = self.embedding_path.is_dir()
        self.is_h5 = self.embedding_path.is_file() and self.embedding_path.suffix == '.h5'
        
        if not (self.is_lmdb or self.is_h5):
            raise ValueError(
                f"Embedding path must be HDF5 file (.h5) or LMDB directory. "
                f"Got: {embedding_path}"
            )
        
        if self.is_lmdb:
            self._init_lmdb()
        else:
            self._init_h5()
    
    def _init_lmdb(self):
        """Initialize LMDB environment."""
        self.env = lmdb.open(
            str(self.embedding_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.entries = self.env.stat().get("entries", 0)
        logger.info(f"Opened LMDB database at: {self.embedding_path}")
    
    def _init_h5(self):
        """Initialize HDF5 file handle."""
        self.h5_file = h5py.File(self.embedding_path, 'r')
        self.entries = len(self.h5_file.keys())
        logger.info(f"Opened HDF5 file at: {self.embedding_path}")
    
    def __len__(self) -> int:
        return getattr(self, "entries", 0)
    
    def _fetch_from_lmdb(self, txn, key: str) -> Optional[np.ndarray]:
        """Fetch embedding bytes from LMDB within an existing transaction."""
        # Try full key first (LMDB might store full HDF5-style keys)
        key_bytes = key.encode('utf-8')
        value = txn.get(key_bytes)
        
        # If not found, try normalized HDF5 key format
        if value is None:
            normalized_key = normalize_h5_key(key)
            key_bytes = normalized_key.encode('utf-8')
            value = txn.get(key_bytes)
        
        # If still not found, try domain ID extraction
        if value is None:
            try:
                domain_id = extract_domain_id(key)
                key_bytes = domain_id.encode('utf-8')
                value = txn.get(key_bytes)
            except ValueError:
                pass
        
        if value is None:
            return None
        
        # Convert bytes to numpy array (float16, 1024 dims)
        # Keep as float16 for memory efficiency; downstream code converts to float32
        # Make writable copy to avoid PyTorch warnings
        return np.frombuffer(value, dtype=np.float16).copy()

    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Get embedding for a given key (opens its own LMDB transaction if needed)."""
        if self.is_lmdb:
            with self.env.begin(write=False) as txn:
                return self._fetch_from_lmdb(txn, key)
        # HDF5 format
        try:
            normalized_key = normalize_h5_key(key)
            return np.array(self.h5_file[normalized_key])
        except KeyError:
            try:
                return np.array(self.h5_file[key])
            except KeyError:
                return None

    def get_embeddings_batch(self, keys: List[str]) -> List[tuple[str, Optional[np.ndarray]]]:
        """Batch fetch embeddings, reusing a single LMDB transaction when applicable."""
        results: List[tuple[str, Optional[np.ndarray]]] = []
        if self.is_lmdb:
            with self.env.begin(write=False) as txn:
                for key in keys:
                    results.append((key, self._fetch_from_lmdb(txn, key)))
        else:
            for key in keys:
                results.append((key, self.get_embedding(key)))
        return results
    
    def iter_embeddings(
        self,
        keys: Optional[List[str]] = None,
        batch_size: int = 1024,
        to_float32: bool = False,
    ):
        """Yield batches of embeddings for streaming workflows.
        
        Args:
            keys: Optional subset of keys to iterate. If None, iterate all keys
                  in storage (LMDB cursor or H5 keys).
            batch_size: Number of items per yielded batch.
            to_float32: Convert embeddings to float32 before yielding.
        Yields:
            List of (key, embedding or None) pairs.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        
        if self.is_lmdb:
            # Full scan over LMDB
            if keys is None:
                with self.env.begin(write=False) as txn:
                    cursor = txn.cursor()
                    batch = []
                    for key_bytes, value in cursor:
                        emb = np.frombuffer(value, dtype=np.float16).copy()
                        if to_float32:
                            emb = emb.astype(np.float32, copy=False)
                        batch.append((key_bytes.decode("utf-8"), emb))
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                    if batch:
                        yield batch
            else:
                for i in range(0, len(keys), batch_size):
                    chunk_keys = keys[i : i + batch_size]
                    batch = []
                    with self.env.begin(write=False) as txn:
                        for key in chunk_keys:
                            emb = self._fetch_from_lmdb(txn, key)
                            if emb is not None and to_float32:
                                emb = emb.astype(np.float32, copy=False)
                            batch.append((key, emb))
                    yield batch
        else:
            # H5 iteration
            h5_keys = list(self.h5_file.keys()) if keys is None else keys
            for i in range(0, len(h5_keys), batch_size):
                batch = []
                for key in h5_keys[i : i + batch_size]:
                    emb = self.get_embedding(key)
                    if emb is not None and to_float32:
                        emb = emb.astype(np.float32, copy=False)
                    batch.append((key, emb))
                yield batch
    
    def close(self):
        """Close the embedding storage."""
        if self.is_lmdb:
            self.env.close()
        else:
            self.h5_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
