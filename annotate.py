"""Annotate protein sequences using k-NN search in vector database."""

import hydra
from omegaconf import DictConfig
import torch
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import logging
import time
import warnings

from contrasted.utils import load_h5_keys_from_fasta, load_labels, extract_domain_id
from contrasted.model import CathSupConModel
from contrasted.faiss_utils import search_faiss_index
import faiss

logger = logging.getLogger(__name__)


def compute_consensus(
    neighbor_annotations: list[str],
    distances: np.ndarray,
    distance_cutoff: float,
) -> tuple[str, float]:
    """Compute consensus annotation from k-nearest neighbors.
    
    Args:
        neighbor_annotations: List of annotations for k neighbors
        distances: Cosine distances (1 - similarity) for k neighbors
        distance_cutoff: Maximum distance to consider valid
        
    Returns:
        predicted_annotation: Consensus annotation (or 'unknown')
        confidence: Confidence score [0, 1]
    """
    valid_mask = distances <= distance_cutoff
    valid_annotations = [
        ann for ann, valid in zip(neighbor_annotations, valid_mask) if valid
    ]
    
    if not valid_annotations:
        return "unknown", 0.0
    
    counts = Counter(valid_annotations)
    predicted_annotation, count = counts.most_common(1)[0]
    confidence = count / len(valid_annotations)
    
    return predicted_annotation, confidence


@torch.no_grad()
def annotate_sequences(
    model: CathSupConModel,
    h5_file: h5py.File,
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
    batch_size: int = 2048,
) -> list[dict]:
    """Annotate query sequences using k-NN search.
    
    Args:
        model: Trained CathSupConModel
        h5_file: Open HDF5 file with query embeddings
        h5_keys: List of HDF5 keys to annotate
        index: FAISS index
        ref_domain_ids: Domain IDs corresponding to index vectors
        id_to_annotation: Mapping from domain_id to annotation index
        idx_to_annotation: Mapping from annotation index to annotation string
        device: Device for inference
        k: Number of nearest neighbors
        distance_cutoff: Maximum distance for valid predictions
        return_distance: Include distances in output
        return_confidence: Include confidence scores in output
        return_true_annotation: Include true annotations in output
        batch_size: Batch size for processing
        
    Returns:
        List of annotation results (dicts)
    """
    model.eval()
    results = []
    
    for i in tqdm(range(0, len(h5_keys), batch_size), desc="Annotating sequences"):
        batch_keys = h5_keys[i : i + batch_size]
        batch_embs = []
        batch_query_ids = []
        
        for h5_key in batch_keys:
            try:
                embedding = torch.from_numpy(h5_file[h5_key][:]).float()
                batch_embs.append(embedding)
                batch_query_ids.append(extract_domain_id(h5_key))
            except KeyError:
                logger.warning(f"Missing embedding for key: {h5_key}")
                query_id = extract_domain_id(h5_key)
                result = {"query_id": query_id, "predicted_annotation": "missing_embedding"}
                if return_true_annotation:
                    true_idx = id_to_annotation.get(query_id, -1)
                    result["true_annotation"] = idx_to_annotation.get(true_idx, "unknown")
                results.append(result)
                continue
        
        if not batch_embs:
            continue
        
        # Project embeddings
        batch_tensor = torch.stack(batch_embs).to(device)
        projected = model(batch_tensor)
        query_vectors = projected.cpu().numpy().astype(np.float32)
        
        # Search index
        similarities, indices = search_faiss_index(index, query_vectors, k)
        distances = 1.0 - similarities  # Convert similarity to distance
        
        # Process results
        for query_id, query_distances, query_indices in zip(
            batch_query_ids, distances, indices
        ):
            neighbor_domain_ids = ref_domain_ids[query_indices]
            neighbor_annotations = [
                idx_to_annotation.get(id_to_annotation.get(nid, -1), "unknown")
                for nid in neighbor_domain_ids
            ]
            
            predicted_annotation, confidence = compute_consensus(
                neighbor_annotations, query_distances, distance_cutoff
            )
            
            result = {
                "query_id": query_id,
                "predicted_annotation": predicted_annotation,
            }
            
            if return_true_annotation:
                true_idx = id_to_annotation.get(query_id, -1)
                result["true_annotation"] = idx_to_annotation.get(true_idx, "unknown")
            
            if return_distance:
                result["distance"] = float(query_distances[0])
            
            if return_confidence:
                result["confidence"] = float(confidence)
            
            results.append(result)
    
    return results


@hydra.main(version_base=None, config_path="configs", config_name="annotate")
def main(cfg: DictConfig):
    """Annotate protein sequences using k-NN search."""
    
    # Setup device with auto-detection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Validate inputs
    input_path = Path(cfg.input)
    model_path = Path(cfg.model_path)
    index_path = Path(cfg.index)
    ids_path = index_path.with_suffix(".npy")
    annotation_path = Path(cfg.id_to_annotation)
    
    for path, name in [
        (input_path, "Input FASTA"),
        (model_path, "Model checkpoint"),
        (index_path, "FAISS index"),
        (ids_path, "Domain IDs file"),
        (annotation_path, "Annotation file"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = CathSupConModel.load_from_checkpoint(str(model_path), strict=False)
    model.eval()
    model.to(device)
    
    # Load FAISS index and domain IDs
    logger.info(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(str(index_path))
    ref_domain_ids = np.load(ids_path)
    logger.info(f"Loaded index with {index.ntotal} vectors")
    
    # Load annotations
    logger.info(f"Loading annotations from: {annotation_path}")
    id_to_annotation, idx_to_annotation = load_labels(annotation_path)
    logger.info(f"Loaded {len(idx_to_annotation)} annotation classes")
    
    # Load query sequences
    logger.info(f"Loading query sequences from: {input_path}")
    h5_keys = load_h5_keys_from_fasta(input_path)
    logger.info(f"Processing {len(h5_keys)} query sequences")
    
    # Annotate sequences
    embedding_file = Path(cfg.get("embedding_file", "data/cath-domain-seqs-S100.h5"))
    logger.info(f"Loading embeddings from: {embedding_file}")
    
    start_time = time.time()
    with h5py.File(embedding_file, "r") as h5f:
        results = annotate_sequences(
            model=model,
            h5_file=h5f,
            h5_keys=h5_keys,
            index=index,
            ref_domain_ids=ref_domain_ids,
            id_to_annotation=id_to_annotation,
            idx_to_annotation=idx_to_annotation,
            device=device,
            k=cfg.k,
            distance_cutoff=cfg.distance_cutoff,
            return_distance=cfg.return_distance,
            return_confidence=cfg.return_confidence,
            return_true_annotation=cfg.get("return_true_annotation", True),
            batch_size=2048,
        )
    annotation_time = time.time() - start_time
    
    # Save results
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved annotations to: {output_path}")
    
    # Print summary statistics
    total = len(results)
    unknown = (df["predicted_annotation"] == "unknown").sum()
    missing = (df["predicted_annotation"] == "missing_embedding").sum()
    annotated = total - unknown - missing
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Annotation Summary:")
    logger.info(f"  Total queries: {total}")
    logger.info(f"  Successfully annotated: {annotated} ({100*annotated/total:.1f}%)")
    logger.info(f"  Unknown (no neighbors within cutoff): {unknown} ({100*unknown/total:.1f}%)")
    logger.info(f"  Missing embeddings: {missing} ({100*missing/total:.1f}%)")
    logger.info(
        f"  Annotation time: {annotation_time:.2f}s "
        f"({annotation_time/total*1000:.2f}ms per query)"
    )
    
    if cfg.return_confidence and annotated > 0:
        valid_conf = df[
            ~df["predicted_annotation"].isin(["unknown", "missing_embedding"])
        ]["confidence"]
        logger.info(f"  Mean confidence: {valid_conf.mean():.3f}")
        logger.info(f"  Median confidence: {valid_conf.median():.3f}")
    
    # Compute metrics if true annotations are available
    if cfg.get("return_true_annotation", True) and "true_annotation" in df.columns:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        
        valid_mask = ~df["predicted_annotation"].isin(["unknown", "missing_embedding"])
        valid_df = df[valid_mask]
        
        if len(valid_df) > 0:
            y_true = valid_df["true_annotation"].values
            y_pred = valid_df["predicted_annotation"].values
            all_classes = sorted(set(y_true) | set(y_pred))
            
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="y_pred contains classes not in y_true"
                )
                accuracy = accuracy_score(y_true, y_pred)
                balanced_acc = balanced_accuracy_score(y_true, y_pred)
                macro_f1 = f1_score(
                    y_true, y_pred, average="macro", zero_division=0, labels=all_classes
                )
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Performance Metrics (on {len(valid_df)} valid predictions):")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
            logger.info(f"  Macro F1: {macro_f1:.4f}")
            
            df["correct"] = df["predicted_annotation"] == df["true_annotation"]
            correct_count = valid_df[
                valid_df["predicted_annotation"] == valid_df["true_annotation"]
            ].shape[0]
            logger.info(
                f"  Correct predictions: {correct_count}/{len(valid_df)} "
                f"({100*correct_count/len(valid_df):.1f}%)"
            )
            
            df.to_csv(output_path, sep="\t", index=False)
    
    logger.info(f"{'='*50}\n")
    logger.info("✓ Annotation complete!")


if __name__ == "__main__":
    main()

