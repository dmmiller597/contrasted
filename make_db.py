"""Create a vector database from trained model embeddings."""

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from contrasted.data import load_domain_ids_from_fasta
from contrasted.model import ContrastiveModel
from contrasted.search import VectorIndex
from contrasted.utils import load_labels

logger = logging.getLogger(__name__)


def load_embeddings_pt(
    pt_path: Path,
    domain_ids: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """Load embeddings from .pt file for specified domain IDs."""
    logger.info(f"Loading embeddings from: {pt_path}")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"].float()
    ids = data["ids"]

    id_to_idx = {id_: i for i, id_ in enumerate(ids)}

    found_indices: list[int] = []
    found_ids = []
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
        raise ValueError(f"No embeddings found in {pt_path}")

    return embeddings[found_indices], found_ids


@torch.no_grad()
def project_embeddings(
    model: ContrastiveModel,
    embeddings: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Project embeddings through trained model."""
    model.eval()
    projected = []

    for i in tqdm(range(0, len(embeddings), batch_size), desc="Projecting"):
        batch = embeddings[i : i + batch_size].to(device)
        proj = model(batch).cpu()
        projected.append(proj)

    return torch.cat(projected, dim=0)


@hydra.main(version_base=None, config_path="configs", config_name="make_db")
def main(cfg: DictConfig):
    """Create vector index from trained model embeddings."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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

    embedding_file = Path(cfg.get("embedding_file", "data/cath-c123-S100.pt"))
    raw_embeddings, domain_ids = load_embeddings_pt(embedding_file, domain_ids)

    projected = project_embeddings(
        model,
        raw_embeddings,
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
    index = VectorIndex(projected, domain_ids, labels=labels, dtype=dtype)

    index_path = Path(cfg.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)


if __name__ == "__main__":
    main()
