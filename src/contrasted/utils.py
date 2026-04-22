"""Utility helpers for contrasted."""

from pathlib import Path

import lightning as L
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed every relevant RNG and optionally enable deterministic kernels.

    Wraps :func:`lightning.pytorch.seed_everything` and adds
    :func:`torch.use_deterministic_algorithms` so CUBLAS kernels raise when
    they cannot run deterministically (rather than silently using a
    non-deterministic path).
    """
    resolved = L.seed_everything(seed, workers=True)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return resolved


def load_labels(
    label_path: str | Path,
) -> tuple[dict[str, int], dict[int, str]]:
    """Load domain labels from a two-column whitespace-delimited text file.

    Lines starting with ``#`` are ignored. Columns: ``domain_id`` then
    ``superfamily``. Returns ``(id_to_sf_idx, idx_to_sf)``.
    """
    id_to_sf_idx: dict[str, int] = {}
    sf_to_idx: dict[str, int] = {}
    with open(label_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            domain_id, superfamily = parts[0], parts[1]
            if superfamily not in sf_to_idx:
                sf_to_idx[superfamily] = len(sf_to_idx)
            id_to_sf_idx[domain_id] = sf_to_idx[superfamily]
    return id_to_sf_idx, {v: k for k, v in sf_to_idx.items()}


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def accuracy(preds: torch.Tensor, target: torch.Tensor) -> float:
    """Fraction of predictions that match ``target``.

    Comparison runs on whichever device ``preds`` and ``target`` share;
    only the final scalar is moved to host memory.
    """
    if preds.numel() == 0:
        return 0.0
    return (preds == target.to(preds.device)).float().mean().item()
