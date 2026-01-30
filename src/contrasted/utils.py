"""Utility functions for contrasted."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_labels(label_path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load domain labels from a two-column text file.

    Expected format (whitespace-delimited, comments allowed):
        domain_id superfamily
    """
    id_to_sf_idx: dict[str, int] = {}
    sf_to_idx: dict[str, int] = {}

    with open(label_path) as f:
        for line in f:
            if not line.startswith("#"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    domain_id, superfamily = parts[0], parts[1]
                    if superfamily not in sf_to_idx:
                        sf_to_idx[superfamily] = len(sf_to_idx)
                    id_to_sf_idx[domain_id] = sf_to_idx[superfamily]

    return id_to_sf_idx, {v: k for k, v in sf_to_idx.items()}


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
