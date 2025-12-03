"""Annotate protein sequences using k-NN search in vector database."""

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from typing import List, Dict
import logging
import time
import warnings

from contrasted.utils import load_h5_keys_from_fasta, load_labels, extract_domain_id, resolve_fasta_paths, EmbeddingReader
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


def compute_consensus(
    neighbor_annotations: list[str],
    distances: np.ndarray,
    distance_cutoff: float,
) -> tuple[str, float]:
    """Compute consensus annotation from k-nearest neighbors."""
    valid_annotations = [
        ann for ann, dist in zip(neighbor_annotations, distances) if dist <= distance_cutoff
    ]
    if not valid_annotations:
        return "unknown", 0.0
    
    predicted, count = Counter(valid_annotations).most_common(1)[0]
    return predicted, count / len(valid_annotations)


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
    batch_size: int = 2048,
) -> list[dict]:
    """Annotate query sequences using k-NN search."""
    model.eval()
    results = []
    
    for i in tqdm(range(0, len(h5_keys), batch_size), desc="Annotating sequences"):
        batch_keys = h5_keys[i : i + batch_size]
        batch_embs = []
        batch_query_ids = []
        
        for h5_key in batch_keys:
            embedding = embedding_reader.get_embedding(h5_key)
            if embedding is None:
                logger.warning(f"Missing embedding for key: {h5_key}")
                query_id = extract_domain_id(h5_key)
                result = {"query_id": query_id, "predicted_annotation": "missing_embedding"}
                if return_true_annotation:
                    true_idx = id_to_annotation.get(query_id, -1)
                    result["true_annotation"] = idx_to_annotation.get(true_idx, "unknown")
                results.append(result)
                continue
            
            batch_embs.append(torch.from_numpy(embedding).float())
            batch_query_ids.append(extract_domain_id(h5_key))
        
        if not batch_embs:
            continue
        
        batch_tensor = torch.stack(batch_embs).to(device)
        query_vectors = model(batch_tensor).cpu().numpy().astype(np.float32)
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


def resolve_input_paths(input_path: Path) -> Dict[str, Path]:
    """Resolve input FASTA paths with error handling."""
    result = resolve_fasta_paths(input_path)
    if not result:
        raise FileNotFoundError(f"No FASTA files found at: {input_path}")
    return result


def process_and_save_results(
    results: List[dict],
    output_path: Path,
    input_name: str,
    annotation_time: float,
    return_confidence: bool,
    return_true_annotation: bool,
):
    """Save results and log summary statistics."""
    df = pd.DataFrame(results)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved annotations to: {output_path}")
    
    total = len(results)
    unknown = (df["predicted_annotation"] == "unknown").sum()
    missing = (df["predicted_annotation"] == "missing_embedding").sum()
    annotated = total - unknown - missing
    
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
    annotation_path = Path(cfg.id_to_annotation)
    
    input_paths = resolve_input_paths(input_path)
    
    for path, name in [(model_path, "Model checkpoint"), (index_path, "FAISS index"),
                       (ids_path, "Domain IDs file"), (annotation_path, "Annotation file")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    
    logger.info(f"Loading model from: {model_path}")
    model = CathSupConModel.load_from_checkpoint(str(model_path), strict=False).eval().to(device)
    
    logger.info(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(str(index_path))
    ref_domain_ids = np.load(ids_path)
    logger.info(f"Loaded index with {index.ntotal} vectors")
    
    logger.info(f"Loading annotations from: {annotation_path}")
    id_to_annotation, idx_to_annotation = load_labels(annotation_path)
    logger.info(f"Loaded {len(idx_to_annotation)} annotation classes")
    
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
            results = annotate_sequences(
                model, embedding_reader, h5_keys, index, ref_domain_ids, id_to_annotation, idx_to_annotation,
                device, cfg.k, cfg.distance_cutoff, cfg.return_distance, cfg.return_confidence,
                cfg.get("return_true_annotation", True), 2048
            )
            
            process_and_save_results(
                results, output_dir / f"{input_name}_annotations.tsv", input_name,
                time.time() - start_time, cfg.return_confidence, cfg.get("return_true_annotation", True)
            )
    
    logger.info(f"\n{'='*70}")
    logger.info("✓ All annotations complete!")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
