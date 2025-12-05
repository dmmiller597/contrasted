"""Annotate protein sequences using k-NN search in vector database."""

import csv
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import time

from contrasted.utils import (
    load_h5_keys_from_fasta,
    load_labels,
    extract_domain_id,
    resolve_fasta_paths,
    EmbeddingReader,
)
from contrasted.model import CathSupConModel
from contrasted.faiss_utils import search_faiss_index
import faiss

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    """Configure logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "annotate.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


@torch.no_grad()
def annotate_sequences(
    model: CathSupConModel,
    embedding_reader: EmbeddingReader,
    h5_keys: list[str],
    index: faiss.Index,
    ref_domain_ids: np.ndarray,
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
) -> dict:
    """Annotate query sequences using k-NN search and stream results to disk."""
    model.eval()
    flush_every = 50000  # fixed flush cadence to avoid config bloat
    
    # Precompute reference annotations aligned to FAISS ids for O(1) lookup.
    ref_annotation_idx = np.array([id_to_annotation.get(rid, -1) for rid in ref_domain_ids], dtype=np.int64)
    
    headers = ["query_id", "predicted_annotation"]
    if return_true_annotation:
        headers.append("true_annotation")
    if return_distance:
        headers.append("distance")
    if return_confidence:
        headers.append("confidence")
    
    total = unknown = missing = annotated = 0
    missing_batch_count = 0
    
    confidences = [] if return_confidence else None
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(headers)
        
        for i in tqdm(range(0, len(h5_keys), batch_size), desc="Annotating sequences"):
            batch_keys = h5_keys[i : i + batch_size]
            batch_embs = []
            batch_query_ids = []
            
            batch_results = []
            # Reuse one LMDB transaction per batch when applicable.
            batch_embeddings = (
                embedding_reader.get_embeddings_batch(batch_keys)
                if getattr(embedding_reader, "is_lmdb", False)
                else [(k, embedding_reader.get_embedding(k)) for k in batch_keys]
            )
            
            for h5_key, embedding in batch_embeddings:
                query_id = extract_domain_id(h5_key)
                if embedding is None:
                    missing += 1
                    missing_batch_count += 1
                    result = {"query_id": query_id, "predicted_annotation": "missing_embedding"}
                    if return_true_annotation:
                        true_idx = id_to_annotation.get(query_id, -1)
                        result["true_annotation"] = idx_to_annotation.get(true_idx, "unknown")
                    batch_results.append(result)
                    continue
                
                batch_embs.append(torch.from_numpy(embedding).float())
                batch_query_ids.append(query_id)
            
            if missing_batch_count:
                logger.debug(f"Missing {missing_batch_count} embeddings in current batch.")
                missing_batch_count = 0
            
            if batch_embs:
                batch_tensor = torch.stack(batch_embs).to(device)
                query_vectors = model(batch_tensor).cpu().numpy().astype(np.float32)
                similarities, indices = search_faiss_index(index, query_vectors, k)
                distances = 1.0 - similarities
                
                for query_id, query_distances, query_indices in zip(batch_query_ids, distances, indices):
                    # k=1 expected, but handle general k.
                    best_idx = query_indices[0]
                    best_distance = query_distances[0]
                    
                    if best_distance > distance_cutoff:
                        predicted_annotation = "unknown"
                        confidence = 0.0
                    else:
                        ann_idx = ref_annotation_idx[best_idx]
                        predicted_annotation = idx_to_annotation.get(ann_idx, "unknown")
                        confidence = 1.0 if k == 1 else float(np.mean(query_distances <= distance_cutoff))
                    
                    result = {"query_id": query_id, "predicted_annotation": predicted_annotation}
                    
                    if return_true_annotation:
                        true_annotation = idx_to_annotation.get(id_to_annotation.get(query_id, -1), "unknown")
                        result["true_annotation"] = true_annotation
                    if return_distance:
                        result["distance"] = float(best_distance)
                    if return_confidence:
                        result["confidence"] = float(confidence)
                        if predicted_annotation not in {"unknown", "missing_embedding"}:
                            confidences.append(float(confidence))
                    
                    batch_results.append(result)
            
            # Stream out results for this batch.
            for res in batch_results:
                row = [res["query_id"], res["predicted_annotation"]]
                if return_true_annotation:
                    row.append(res.get("true_annotation", "unknown"))
                if return_distance:
                    row.append(res.get("distance", ""))
                if return_confidence:
                    row.append(res.get("confidence", ""))
                writer.writerow(row)
            
            if (i // batch_size) % max(1, flush_every // max(1, batch_size)) == 0:
                f.flush()
            
            total += len(batch_results)
            batch_unknown = sum(1 for r in batch_results if r["predicted_annotation"] == "unknown")
            batch_missing = sum(1 for r in batch_results if r["predicted_annotation"] == "missing_embedding")
            unknown += batch_unknown
            annotated += len(batch_results) - batch_unknown - batch_missing
    
    return {
        "total": total,
        "unknown": unknown,
        "missing": missing,
        "annotated": annotated,
        "confidences": confidences or [],
    }


def resolve_input_paths(input_path: Path) -> Dict[str, Path]:
    """Resolve input FASTA paths with error handling."""
    result = resolve_fasta_paths(input_path)
    if not result:
        raise FileNotFoundError(f"No FASTA files found at: {input_path}")
    return result


def process_and_save_results(
    summary: dict,
    output_path: Path,
    input_name: str,
    annotation_time: float,
    return_confidence: bool,
):
    """Log summary statistics (results already streamed to disk)."""
    total = summary["total"]
    unknown = summary["unknown"]
    missing = summary["missing"]
    annotated = summary["annotated"]
    
    logger.info(f"Saved annotations to: {output_path}")
    logger.info(f"\n{'='*50}")
    logger.info(f"Annotation Summary for {input_name}:")
    logger.info(f"  Total queries: {total}")
    logger.info(f"  Successfully annotated: {annotated} ({100*annotated/total:.1f}%)")
    logger.info(f"  Unknown (no neighbors within cutoff): {unknown} ({100*unknown/total:.1f}%)")
    logger.info(f"  Missing embeddings: {missing} ({100*missing/total:.1f}%)")
    logger.info(
        f"  Annotation time: {annotation_time:.2f}s "
        f"({annotation_time/total*1000:.2f}ms per query)"
    )
    
    if return_confidence and annotated > 0 and summary["confidences"]:
        conf_arr = np.array(summary["confidences"])
        logger.info(f"  Mean confidence: {conf_arr.mean():.3f}")
        logger.info(f"  Median confidence: {np.median(conf_arr):.3f}")
    
    logger.info(f"{'='*50}\n")


@hydra.main(version_base=None, config_path="configs", config_name="annotate")
def main(cfg: DictConfig):
    """Annotate protein sequences using k-NN search."""
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    input_path = Path(cfg.input)
    model_path = Path(cfg.model_path)
    index_path = Path(cfg.index)
    ids_path = index_path.with_suffix(".npy")
    annotation_path = Path(cfg.id_to_annotation) if cfg.get("id_to_annotation") else None
    
    input_paths = resolve_input_paths(input_path)
    
    required_files = [(model_path, "Model checkpoint"), (index_path, "FAISS index"), (ids_path, "Domain IDs file")]
    for path, name in required_files:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    
    if annotation_path:
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    
    logger.info(f"Loading model from: {model_path}")
    model = CathSupConModel.load_from_checkpoint(str(model_path), strict=False).eval().to(device)
    
    logger.info(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(str(index_path))
    ref_domain_ids = np.load(ids_path)
    logger.info(f"Loaded index with {index.ntotal} vectors")
    
    if annotation_path:
        logger.info(f"Loading annotations from: {annotation_path}")
        id_to_annotation, idx_to_annotation = load_labels(annotation_path)
        logger.info(f"Loaded {len(idx_to_annotation)} annotation classes")
    else:
        logger.info("No annotation file provided; true annotations will be skipped.")
        id_to_annotation, idx_to_annotation = {}, {}
    
    output_dir = Path(cfg.get("output_dir", "outputs/annotations"))
    setup_logging(output_dir)
    
    embedding_file = Path(cfg.get("embedding_file", "data/cath-domain-seqs-S100.h5"))
    logger.info(f"Loading embeddings from: {embedding_file}")
    
    with EmbeddingReader(embedding_file) as embedding_reader:
        for input_name, fasta_path in input_paths.items():
            logger.info(f"\n{'='*70}\nProcessing: {input_name} ({fasta_path})\n{'='*70}")
            
            h5_keys = load_h5_keys_from_fasta(fasta_path)
            logger.info(f"Processing {len(h5_keys)} query sequences")
            
            start_time = time.time()
            output_path = output_dir / f"{input_name}_annotations.tsv"
            return_true_annotation = bool(cfg.get("return_true_annotation", True) and annotation_path)
            return_confidence = bool(cfg.get("return_confidence", False))
            
            results_summary = annotate_sequences(
                model, embedding_reader, h5_keys, index, ref_domain_ids, id_to_annotation, idx_to_annotation,
                device, cfg.k, cfg.distance_cutoff, cfg.return_distance, return_confidence,
                return_true_annotation, output_path,
                cfg.get("batch_size", 2048)
            )
            
            process_and_save_results(
                results_summary, output_path, input_name,
                time.time() - start_time, return_confidence
            )
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ All annotations complete!")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
