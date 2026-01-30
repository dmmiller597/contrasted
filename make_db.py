"""Create a vector database from trained model embeddings."""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from contrasted.data import (
    load_domain_ids_from_fasta,
    load_embedding_dir,
    load_id_to_idx,
)
from contrasted.model import ContrastiveModel
from contrasted.search import FaissIndex, VectorIndex
from contrasted.utils import get_device, load_labels

logger = logging.getLogger(__name__)


def resolve_embedding_indices(
    embedding_dir: Path,
    domain_ids: list[str],
) -> tuple[np.ndarray, list[str], list[int]]:
    """Load embeddings from embedding dir for specified domain IDs."""
    logger.info(f"Loading embeddings from: {embedding_dir}")

    embeddings, ids, _, _ = load_embedding_dir(embedding_dir)
    id_to_idx = load_id_to_idx(embedding_dir, ids)

    found_indices: list[int] = []
    found_ids: list[str] = []
    missing = 0

    for domain_id in domain_ids:
        if domain_id in id_to_idx:
            idx = id_to_idx[domain_id]
            found_indices.append(idx)
            found_ids.append(domain_id)
        else:
            missing += 1

    if missing > 0:
        logger.warning(f"Missing {missing} embeddings")

    if not found_indices:
        raise ValueError(f"No embeddings found in {embedding_dir}")

    return embeddings, found_ids, found_indices


@torch.no_grad()
def project_embeddings(
    model: ContrastiveModel,
    embeddings: np.ndarray,
    indices: list[int],
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Project embeddings through trained model."""
    model.eval()
    projected = []

    for i in tqdm(range(0, len(indices), batch_size), desc="Projecting"):
        batch_indices = indices[i : i + batch_size]
        batch_np = np.asarray(embeddings[batch_indices])
        batch = torch.from_numpy(batch_np).float().to(device)
        proj = model(batch).cpu()
        projected.append(proj)

    return torch.cat(projected, dim=0)


@hydra.main(version_base=None, config_path="configs", config_name="make_db")
def main(cfg: DictConfig):
    """Create vector index from trained model embeddings."""
    device = get_device()
    logger.info(f"Using device: {device}")

    input_path = Path(cfg.input)
    model_path = Path(cfg.model_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    model = ContrastiveModel.load_from_checkpoint(
        str(model_path), strict=False, weights_only=False
    )
    model.eval()
    model.to(device)

    logger.info(f"Loading sequences from: {input_path}")
    domain_ids = load_domain_ids_from_fasta(input_path)

    if cfg.ids:
        ids_set = set(cfg.ids)
        domain_ids = [d for d in domain_ids if d in ids_set]
        logger.info(f"Filtered to {len(domain_ids)} sequences matching provided IDs")
    else:
        logger.info(f"Processing {len(domain_ids)} sequences")

    embedding_dir = Path(cfg.get("embedding_dir", "data/cath-c123-S100"))
    raw_embeddings, domain_ids, indices = resolve_embedding_indices(
        embedding_dir,
        domain_ids,
    )

    projected = project_embeddings(
        model,
        raw_embeddings,
        indices,
        device,
        batch_size=cfg.get("project_batch_size", 4096),
    )

    logger.info(
        f"Generated {projected.shape[0]} embeddings of dimension {projected.shape[1]}"
    )

    label_path = Path(cfg.label_file) if cfg.get("label_file") else None
    labels = None
    if label_path:
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        id_to_label, idx_to_label = load_labels(label_path)
        labels = [
            idx_to_label.get(id_to_label.get(domain_id, -1), "unknown")
            for domain_id in domain_ids
        ]
        logger.info(f"Loaded labels for {len(labels)} domains")

    dtype = torch.float16 if cfg.get("dtype", "float16") == "float16" else torch.float32
    index_backend = str(cfg.get("index_backend", "faiss")).lower()
    if index_backend == "faiss":
        index = FaissIndex(projected, domain_ids, labels=labels)
    else:
        index = VectorIndex(projected, domain_ids, labels=labels, dtype=dtype)

    index_path = Path(cfg.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)


if __name__ == "__main__":
    main()
