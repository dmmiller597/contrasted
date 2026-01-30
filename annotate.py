"""Annotate protein sequences using k-NN search in vector database."""

import csv
import logging
import time
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
    resolve_fasta_paths,
)
from contrasted.model import ContrastiveModel
from contrasted.search import FaissIndex, VectorIndex
from contrasted.utils import get_device, load_labels

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Simple embedding store using embedding directory."""

    def __init__(self, embedding_dir: Path):
        logger.info(f"Loading embeddings from: {embedding_dir}")
        embeddings, ids, _, _ = load_embedding_dir(embedding_dir)
        self.embeddings = embeddings
        self.ids = ids
        self.id_to_idx = load_id_to_idx(embedding_dir, ids)
        logger.info(f"Loaded {len(self.ids)} embeddings")

    def get_batch(
        self,
        domain_ids: list[str],
    ) -> tuple[torch.Tensor, list[str], list[str]]:
        """Get embeddings for a batch of domain IDs.

        Returns:
            embeddings: Tensor of found embeddings
            found_ids: List of IDs that were found
            missing_ids: List of IDs that were not found
        """
        found_indices = []
        found_ids = []
        missing_ids = []

        for domain_id in domain_ids:
            if domain_id in self.id_to_idx:
                found_indices.append(self.id_to_idx[domain_id])
                found_ids.append(domain_id)
            else:
                missing_ids.append(domain_id)

        if found_indices:
            embeddings_np = np.asarray(self.embeddings[found_indices])
            embeddings = torch.from_numpy(embeddings_np).float()
        else:
            embeddings = torch.empty(0, self.embeddings.shape[1])

        return embeddings, found_ids, missing_ids


@torch.no_grad()
def annotate_sequences(
    model: ContrastiveModel,
    store: EmbeddingStore,
    domain_ids: list[str],
    index: VectorIndex,
    id_to_annotation: dict[str, int],
    idx_to_annotation: dict[int, str],
    device: torch.device,
    k: int,
    distance_cutoff: float,
    return_distance: bool,
    return_confidence: bool,
    return_true_annotation: bool,
    output_path: Path,
    batch_size: int = 2048,
    search_chunk_size: int | None = None,
) -> dict:
    """Annotate query sequences using k-NN search."""
    model.eval()

    headers = ["query_id", "predicted_annotation"]
    if return_true_annotation:
        headers.append("true_annotation")
    if return_distance:
        headers.append("distance")
    if return_confidence:
        headers.append("confidence")

    total = unknown = missing = annotated = 0
    confidences = [] if return_confidence else None

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)

        for i in tqdm(range(0, len(domain_ids), batch_size), desc="Annotating"):
            batch_ids = domain_ids[i : i + batch_size]
            batch_results = []

            embeddings, found_ids, missing_ids = store.get_batch(batch_ids)

            for domain_id in missing_ids:
                missing += 1
                result = {
                    "query_id": domain_id,
                    "predicted_annotation": "missing_embedding",
                }
                if return_true_annotation:
                    true_idx = id_to_annotation.get(domain_id, -1)
                    result["true_annotation"] = idx_to_annotation.get(
                        true_idx, "unknown"
                    )
                batch_results.append(result)

            if len(found_ids) > 0:
                query_vectors = model(embeddings.to(device))
                similarities, indices = index.search(
                    query_vectors,
                    k,
                    chunk_size=search_chunk_size,
                )
                distances = 1.0 - similarities.cpu()

                for j, query_id in enumerate(found_ids):
                    best_idx = indices[j, 0].item()
                    best_distance = distances[j, 0].item()

                    if best_distance > distance_cutoff:
                        predicted_annotation = "unknown"
                        confidence = 0.0
                    else:
                        if index.labels is not None:
                            predicted_annotation = index.labels[best_idx]
                        else:
                            ref_id = index.ids[best_idx]
                            ann_idx = id_to_annotation.get(ref_id, -1)
                            predicted_annotation = idx_to_annotation.get(
                                ann_idx, "unknown"
                            )
                        confidence = (
                            1.0
                            if k == 1
                            else float((distances[j] <= distance_cutoff).float().mean())
                        )

                    result = {
                        "query_id": query_id,
                        "predicted_annotation": predicted_annotation,
                    }

                    if return_true_annotation:
                        true_annotation = idx_to_annotation.get(
                            id_to_annotation.get(query_id, -1), "unknown"
                        )
                        result["true_annotation"] = true_annotation
                    if return_distance:
                        result["distance"] = float(best_distance)
                    if return_confidence:
                        result["confidence"] = float(confidence)
                        if predicted_annotation not in {"unknown", "missing_embedding"}:
                            confidences.append(float(confidence))

                    batch_results.append(result)

            for res in batch_results:
                row = [res["query_id"], res["predicted_annotation"]]
                if return_true_annotation:
                    row.append(res.get("true_annotation", "unknown"))
                if return_distance:
                    row.append(res.get("distance", ""))
                if return_confidence:
                    row.append(res.get("confidence", ""))
                writer.writerow(row)

            total += len(batch_results)
            batch_unknown = sum(
                1 for r in batch_results if r["predicted_annotation"] == "unknown"
            )
            batch_missing = sum(
                1
                for r in batch_results
                if r["predicted_annotation"] == "missing_embedding"
            )
            unknown += batch_unknown
            annotated += len(batch_results) - batch_unknown - batch_missing

    return {
        "total": total,
        "unknown": unknown,
        "missing": missing,
        "annotated": annotated,
        "confidences": confidences or [],
    }


@hydra.main(version_base=None, config_path="configs", config_name="annotate")
def main(cfg: DictConfig):
    """Annotate protein sequences using k-NN search."""
    device = get_device()
    logger.info(f"Using device: {device}")

    input_path = Path(cfg.input)
    model_path = Path(cfg.model_path)
    index_path = Path(cfg.index)
    annotation_path = (
        Path(cfg.id_to_annotation) if cfg.get("id_to_annotation") else None
    )

    input_paths = resolve_fasta_paths(input_path)
    if not input_paths:
        raise FileNotFoundError(f"No FASTA files found at: {input_path}")

    for path, name in [
        (model_path, "Model checkpoint"),
        (index_path, "Vector index"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    if annotation_path and not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    logger.info(f"Loading model from: {model_path}")
    model = ContrastiveModel.load_from_checkpoint(
        str(model_path), strict=False, weights_only=False
    )
    model.eval()
    model.to(device)

    logger.info(f"Loading vector index from: {index_path}")
    index_backend = str(cfg.get("index_backend", "faiss")).lower()
    if index_backend == "faiss":
        index = FaissIndex.load(index_path)
    else:
        index = VectorIndex.load(index_path, device=device)
    logger.info(f"Loaded index with {len(index)} vectors")

    if annotation_path:
        logger.info(f"Loading annotations from: {annotation_path}")
        id_to_annotation, idx_to_annotation = load_labels(annotation_path)
        logger.info(f"Loaded {len(idx_to_annotation)} annotation classes")
    else:
        id_to_annotation, idx_to_annotation = {}, {}

    output_dir = Path(cfg.get("output_dir", "outputs/annotations"))
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_dir = Path(cfg.get("embedding_dir", "data/cath-c123-S100"))
    store = EmbeddingStore(embedding_dir)

    for input_name, fasta_path in input_paths.items():
        logger.info(f"Processing: {input_name} ({fasta_path})")

        domain_ids = load_domain_ids_from_fasta(fasta_path)
        logger.info(f"Processing {len(domain_ids)} query sequences")

        start_time = time.time()
        output_path = output_dir / f"{input_name}_annotations.tsv"
        return_true_annotation = bool(
            cfg.get("return_true_annotation", True) and annotation_path
        )
        return_confidence = bool(cfg.get("return_confidence", False))

        summary = annotate_sequences(
            model,
            store,
            domain_ids,
            index,
            id_to_annotation,
            idx_to_annotation,
            device,
            cfg.k,
            cfg.distance_cutoff,
            cfg.return_distance,
            return_confidence,
            return_true_annotation,
            output_path,
            cfg.get("batch_size", 2048),
            cfg.get("search_chunk_size"),
        )

        elapsed = time.time() - start_time
        total = summary["total"]
        annotated = summary["annotated"]
        unknown = summary["unknown"]
        missing = summary["missing"]

        logger.info(f"Saved annotations to: {output_path}")
        logger.info(f"  Total: {total}")
        if total > 0:
            logger.info(f"  Annotated: {annotated} ({100 * annotated / total:.1f}%)")
            logger.info(f"  Unknown: {unknown} ({100 * unknown / total:.1f}%)")
            logger.info(f"  Missing: {missing}")
            logger.info(
                f"  Time: {elapsed:.2f}s ({elapsed / total * 1000:.2f}ms per query)"
            )
        else:
            logger.warning("  No sequences were processed")


if __name__ == "__main__":
    main()
