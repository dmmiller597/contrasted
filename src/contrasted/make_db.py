"""Create a vector database from trained model embeddings."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from configs import hydra_config_dir
from contrasted.data import (
    load_domain_ids_from_fasta,
    resolve_store,
    validate_store_for_projection_head,
)
from contrasted.data_home import locate
from contrasted.projection import ProjectionHead, project
from contrasted.search import VectorIndex
from contrasted.utils import get_device, load_labels, require_exists

logger = logging.getLogger(__name__)


def resolve_labels(
    domain_ids: list[str],
    id_to_label: dict[str, int],
    idx_to_label: dict[int, str],
) -> list[str | None]:
    """Map domain IDs to their label strings; ``None`` for domains without one.

    Unlabeled domains stay type-distinct from real labels — including a genuine
    superfamily literally named ``"unknown"`` — so they can be excluded from
    k-NN voting rather than masquerading as a votable class.
    """
    return [
        idx_to_label[id_to_label[domain_id]] if domain_id in id_to_label else None
        for domain_id in domain_ids
    ]


def run(cfg: DictConfig) -> None:
    """Create vector index from trained model embeddings (Hydra config)."""
    device = get_device()
    logger.info(f"Using device: {device}")

    embedding_dir = cfg.get("embedding_dir")
    if embedding_dir is None or not str(embedding_dir).strip():
        raise ValueError(
            "embedding_dir is required for contrasted-make-db. Pass a store "
            "built by contrasted-build-concat-store or contrasted-embed."
        )

    input_path = Path(cfg.input)
    model_path = locate(cfg.model_path, kind="Projection head")

    require_exists(input_path, "Input FASTA")

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

    store = resolve_store(embedding_dir=embedding_dir)
    validate_store_for_projection_head(
        store,
        input_dim=head.input_dim,
    )
    indices, domain_ids, missing_ids = store.resolve(domain_ids)
    if missing_ids:
        sample = ", ".join(missing_ids[:5])
        more = f" (+{len(missing_ids) - 5} more)" if len(missing_ids) > 5 else ""
        raise ValueError(
            f"{len(missing_ids)} domain IDs not found in {embedding_dir}: "
            f"{sample}{more}"
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

    labels: list[str | None] | None = None
    if label_file := cfg.get("label_file"):
        label_path = require_exists(Path(label_file), "Label file")
        id_to_label, idx_to_label = load_labels(label_path)
        labels = resolve_labels(domain_ids, id_to_label, idx_to_label)
        if n_unlabeled := sum(1 for label in labels if label is None):
            logger.warning(
                f"{n_unlabeled}/{len(labels)} domains have no label in "
                f"{label_path} and will be stored unlabeled (excluded from voting)"
            )
        logger.info(f"Loaded labels for {len(labels) - n_unlabeled} domains")

    if cfg.get("dtype", "float32") == "float16":
        projected = projected.half()
    index = VectorIndex(projected, ids=domain_ids, labels=labels)
    index.save(Path(cfg.index_path))


@hydra.main(version_base=None, config_path=hydra_config_dir(), config_name="make_db")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI wrapper
    run(cfg)


if __name__ == "__main__":
    main()
