"""Create a vector database from trained model embeddings."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from contrasted.data import (
    load_domain_ids_from_fasta,
    resolve_store,
)
from contrasted.embed import build_encode_config
from contrasted.model import ProjectionHead, project
from contrasted.search import VectorIndex
from contrasted.utils import get_device, load_labels

logger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> None:
    """Create vector index from trained model embeddings (Hydra config)."""
    device = get_device()
    logger.info(f"Using device: {device}")

    input_path = Path(cfg.input)
    model_path = Path(cfg.model_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    logger.info(f"Loading projection head from: {model_path}")
    head = ProjectionHead.load(model_path).to(device)
    head.eval()

    logger.info(f"Loading sequences from: {input_path}")
    domain_ids = load_domain_ids_from_fasta(input_path)

    if cfg.ids:
        ids_set = set(cfg.ids)
        domain_ids = [d for d in domain_ids if d in ids_set]
        logger.info(f"Filtered to {len(domain_ids)} sequences matching provided IDs")
    else:
        logger.info(f"Processing {len(domain_ids)} sequences")

    embedding_dir = cfg.get("embedding_dir")
    store = resolve_store(
        embedding_dir=embedding_dir,
        fasta_paths=[input_path],
        encode_config=build_encode_config(cfg.get("embed")),
    )
    indices, domain_ids, missing_ids = store.resolve(domain_ids)
    if missing_ids:
        logger.warning(
            f"{len(missing_ids)} domain IDs not found in "
            f"{embedding_dir or 'on-the-fly encoding'}"
        )
    if not indices:
        raise ValueError("No embeddings found for any requested IDs")

    projected = project(
        head,
        store.embeddings,
        indices,
        device=device,
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
        n_unknown = sum(1 for label in labels if label == "unknown")
        if n_unknown > 0:
            logger.warning(
                f"{n_unknown}/{len(labels)} domains have no label in "
                f"{label_path} and will be labeled 'unknown'"
            )
        logger.info(f"Loaded labels for {len(labels)} domains")

    if cfg.get("dtype", "float32") == "float16":
        projected = projected.half()
    index = VectorIndex(projected, ids=domain_ids, labels=labels)
    index.save(Path(cfg.index_path))


@hydra.main(version_base=None, config_path="pkg://configs", config_name="make_db")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI wrapper
    run(cfg)


if __name__ == "__main__":
    main()
