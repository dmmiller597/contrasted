"""ProstT5 baseline benchmark - k-NN search without model projection.

This script demonstrates the baseline performance using raw ProstT5 embeddings
without projecting through a trained model. It follows the same pipeline as
make_db.py and annotate.py but skips the model projection step.
"""

import argparse
from pathlib import Path
from collections import Counter
import logging
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import h5py
import faiss
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from contrasted.utils import (
    load_h5_keys_from_fasta,
    load_labels,
    extract_domain_id,
    resolve_fasta_paths,
)
from contrasted.faiss_utils import build_faiss_index, search_faiss_index, normalize_embeddings

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path):
    """Configure logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "prostt5_baseline.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_embeddings_directly(
    h5_file: h5py.File,
    h5_keys: List[str],
    batch_size: int = 2048,
) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings directly from HDF5 without model projection.
    
    Args:
        h5_file: Open HDF5 file with embeddings
        h5_keys: List of HDF5 keys to process
        batch_size: Batch size for processing
        
    Returns:
        embeddings: (N, D) normalized embeddings (float32)
        domain_ids: List of domain IDs
    """
    all_embeddings = []
    domain_ids = []
    
    for i in tqdm(range(0, len(h5_keys), batch_size), desc="Loading embeddings"):
        batch_keys = h5_keys[i : i + batch_size]
        batch_embs = []
        
        for h5_key in batch_keys:
            try:
                embedding = h5_file[h5_key][:]
                batch_embs.append(embedding)
                domain_ids.append(extract_domain_id(h5_key))
            except KeyError:
                logger.warning(f"Missing embedding for key: {h5_key}")
                continue
        
        if batch_embs:
            batch_array = np.vstack(batch_embs).astype(np.float32)
            # L2 normalize for cosine similarity
            norms = np.linalg.norm(batch_array, axis=1, keepdims=True)
            batch_array = np.divide(batch_array, norms + 1e-8)
            all_embeddings.append(batch_array)
    
    if not all_embeddings:
        raise ValueError("No valid embeddings loaded")
    
    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    return embeddings_matrix, domain_ids


def compute_consensus(
    neighbor_annotations: List[str],
    distances: np.ndarray,
    distance_cutoff: float,
) -> Tuple[str, float]:
    """Compute consensus annotation from k-nearest neighbors.
    
    Args:
        neighbor_annotations: List of annotations for neighbors
        distances: Array of distances to neighbors
        distance_cutoff: Maximum distance to consider valid
        
    Returns:
        predicted: Predicted annotation class
        confidence: Confidence score (proportion of agreement)
    """
    valid_annotations = [
        ann for ann, dist in zip(neighbor_annotations, distances) if dist <= distance_cutoff
    ]
    if not valid_annotations:
        return "unknown", 0.0
    
    predicted, count = Counter(valid_annotations).most_common(1)[0]
    return predicted, count / len(valid_annotations)


def annotate_sequences(
    h5_file: h5py.File,
    h5_keys: List[str],
    index: faiss.Index,
    ref_domain_ids: np.ndarray,
    id_to_annotation: Dict[str, int],
    idx_to_annotation: Dict[int, str],
    k: int,
    distance_cutoff: float,
    return_distance: bool,
    return_confidence: bool,
    return_true_annotation: bool,
    batch_size: int = 2048,
) -> List[dict]:
    """Annotate query sequences using k-NN search on ProstT5 embeddings.
    
    Args:
        h5_file: Open HDF5 file with embeddings
        h5_keys: List of HDF5 keys to process
        index: FAISS index built from reference embeddings
        ref_domain_ids: Array of reference domain IDs
        id_to_annotation: Mapping from domain ID to annotation index
        idx_to_annotation: Mapping from annotation index to annotation string
        k: Number of nearest neighbors
        distance_cutoff: Maximum distance to consider valid
        return_distance: Whether to return distance to nearest neighbor
        return_confidence: Whether to return confidence score
        return_true_annotation: Whether to return true annotations
        batch_size: Batch size for processing
        
    Returns:
        List of result dictionaries with annotations
    """
    results = []
    
    for i in tqdm(range(0, len(h5_keys), batch_size), desc="Annotating sequences"):
        batch_keys = h5_keys[i : i + batch_size]
        batch_embs = []
        batch_query_ids = []
        
        for h5_key in batch_keys:
            try:
                embedding = h5_file[h5_key][:]
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
        
        # L2 normalize batch embeddings
        batch_array = np.vstack(batch_embs).astype(np.float32)
        norms = np.linalg.norm(batch_array, axis=1, keepdims=True)
        query_vectors = np.divide(batch_array, norms + 1e-8)
        
        similarities, indices = search_faiss_index(index, query_vectors, k)
        distances = 1.0 - similarities
        
        for query_id, query_distances, query_indices in zip(batch_query_ids, distances, indices):
            neighbor_annotations = [
                idx_to_annotation.get(id_to_annotation.get(nid, -1), "unknown")
                for nid in ref_domain_ids[query_indices]
            ]
            
            predicted_annotation, confidence = compute_consensus(
                neighbor_annotations, query_distances, distance_cutoff
            )
            
            result = {"query_id": query_id, "predicted_annotation": predicted_annotation}
            
            if return_true_annotation:
                result["true_annotation"] = idx_to_annotation.get(
                    id_to_annotation.get(query_id, -1), "unknown"
                )
            if return_distance:
                result["distance"] = float(query_distances[0])
            if return_confidence:
                result["confidence"] = float(confidence)
            
            results.append(result)
    
    return results


def process_and_save_results(
    results: List[dict],
    output_path: Path,
    input_name: str,
    annotation_time: float,
    return_confidence: bool,
    return_true_annotation: bool,
) -> None:
    """Save results and log summary statistics.
    
    Args:
        results: List of annotation results
        output_path: Path to save TSV file
        input_name: Name of input dataset
        annotation_time: Total annotation time in seconds
        return_confidence: Whether confidence scores are included
        return_true_annotation: Whether true annotations are included
    """
    df = pd.DataFrame(results)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved annotations to: {output_path}")
    
    total = len(results)
    unknown = (df["predicted_annotation"] == "unknown").sum()
    missing = (df["predicted_annotation"] == "missing_embedding").sum()
    annotated = total - unknown - missing
    
    logger.info(f"\n{'='*50}")
    logger.info(f"ProstT5 Baseline - Annotation Summary for {input_name}:")
    logger.info(f"  Total queries: {total}")
    logger.info(f"  Successfully annotated: {annotated} ({100*annotated/total:.1f}%)")
    logger.info(f"  Unknown (no neighbors within cutoff): {unknown} ({100*unknown/total:.1f}%)")
    logger.info(f"  Missing embeddings: {missing} ({100*missing/total:.1f}%)")
    logger.info(
        f"  Annotation time: {annotation_time:.2f}s "
        f"({annotation_time/total*1000:.2f}ms per query)"
    )
    
    if return_confidence and annotated > 0:
        valid_conf = df[~df["predicted_annotation"].isin(["unknown", "missing_embedding"])]["confidence"]
        logger.info(f"  Mean confidence: {valid_conf.mean():.3f}")
        logger.info(f"  Median confidence: {valid_conf.median():.3f}")
    
    if return_true_annotation and "true_annotation" in df.columns:
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
            correct_count = (valid_df["predicted_annotation"] == valid_df["true_annotation"]).sum()
            logger.info(f"  Correct predictions: {correct_count}/{len(valid_df)} ({100*correct_count/len(valid_df):.1f}%)")
            df.to_csv(output_path, sep="\t", index=False)
    
    logger.info(f"{'='*50}\n")


def main():
    """Run ProstT5 baseline benchmark."""
    parser = argparse.ArgumentParser(
        description="ProstT5 baseline benchmark - k-NN search without model projection"
    )
    parser.add_argument(
        "--train_fasta",
        type=str,
        default="data/clustered_datasets/train.fasta",
        help="Path to training FASTA file",
    )
    parser.add_argument(
        "--test_input",
        type=str,
        default="data/clustered_datasets/test",
        help="Path to test FASTA file or directory",
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        default="data/cath-domain-seqs-S100.h5",
        help="HDF5 file with ProstT5 embeddings",
    )
    parser.add_argument(
        "--id_to_annotation",
        type=str,
        default="data/cath-domain-sf-list.txt",
        help="File mapping domain IDs to CATH superfamily annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/prostt5_baseline",
        help="Directory to save results",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of nearest neighbors to search",
    )
    parser.add_argument(
        "--distance_cutoff",
        type=float,
        default=1.0,
        help="Maximum cosine distance (1 - similarity) to consider valid",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for processing",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    
    # Validate inputs
    train_fasta = Path(args.train_fasta)
    test_input = Path(args.test_input)
    embedding_file = Path(args.embedding_file)
    annotation_file = Path(args.id_to_annotation)
    
    for path, name in [
        (train_fasta, "Train FASTA"),
        (test_input, "Test input"),
        (embedding_file, "Embedding file"),
        (annotation_file, "Annotation file"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    
    logger.info(f"\n{'='*70}")
    logger.info("ProstT5 Baseline Benchmark")
    logger.info(f"{'='*70}")
    logger.info(f"Configuration:")
    logger.info(f"  k = {args.k}")
    logger.info(f"  distance_cutoff = {args.distance_cutoff}")
    logger.info(f"  batch_size = {args.batch_size}")
    
    # Load annotations
    logger.info(f"\nLoading annotations from: {annotation_file}")
    id_to_annotation, idx_to_annotation = load_labels(annotation_file)
    logger.info(f"Loaded {len(idx_to_annotation)} annotation classes")
    
    # Build index from training sequences
    logger.info(f"\nBuilding FAISS index from training sequences...")
    logger.info(f"Train FASTA: {train_fasta}")
    train_h5_keys = load_h5_keys_from_fasta(train_fasta)
    logger.info(f"Processing {len(train_h5_keys)} training sequences")
    
    with h5py.File(embedding_file, "r") as h5f:
        train_embeddings, train_domain_ids = load_embeddings_directly(
            h5f, train_h5_keys, batch_size=args.batch_size
        )
    
    logger.info(f"Generated {train_embeddings.shape[0]} embeddings of dimension {train_embeddings.shape[1]}")
    index = build_faiss_index(train_embeddings)
    logger.info(f"✓ FAISS index built with {index.ntotal} vectors")
    
    # Resolve test inputs
    test_paths = resolve_fasta_paths(test_input)
    if not test_paths:
        raise FileNotFoundError(f"No FASTA files found at: {test_input}")
    
    logger.info(f"\nFound {len(test_paths)} test file(s)")
    
    # Annotate test sequences
    with h5py.File(embedding_file, "r") as h5f:
        for test_name, test_fasta_path in test_paths.items():
            logger.info(f"\n{'='*70}\nProcessing: {test_name} ({test_fasta_path})\n{'='*70}")
            
            test_h5_keys = load_h5_keys_from_fasta(test_fasta_path)
            logger.info(f"Processing {len(test_h5_keys)} test sequences")
            
            start_time = time.time()
            results = annotate_sequences(
                h5f,
                test_h5_keys,
                index,
                np.array(train_domain_ids),
                id_to_annotation,
                idx_to_annotation,
                args.k,
                args.distance_cutoff,
                return_distance=True,
                return_confidence=True,
                return_true_annotation=True,
                batch_size=args.batch_size,
            )
            
            process_and_save_results(
                results,
                output_dir / f"{test_name}_prostt5_annotations.tsv",
                test_name,
                time.time() - start_time,
                return_confidence=True,
                return_true_annotation=True,
            )
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ ProstT5 baseline benchmark complete!")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
