import random
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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


def extract_domain_id(h5_key: str) -> str:
    """Extract domain ID from HDF5 key.
    
    Format: cath|4_4_0|12e8H01_1-113 -> 12e8H01
    """
    return h5_key.split('|')[2].split('_')[0]


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
