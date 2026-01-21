"""Assign CATH superfamilies to TED sequences using nearest neighbor search.

Loads CATH ProtT5 embeddings from H5, projects through trained model, builds FAISS index.
Then projects TED ProtT5 embeddings from LMDB and finds nearest CATH neighbor.
Outputs TSV with TED header, CATH assignment, and cosine distance.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from contrasted.utils import (
    EmbeddingReader,
    extract_domain_id,
    load_labels,
)
from contrasted.model import CathSupConModel
from contrasted.faiss_utils import build_faiss_index, search_faiss_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def project_embeddings(
    model: CathSupConModel,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Project embeddings through trained model and L2-normalize.
    
    Args:
        model: Trained CathSupConModel
        embeddings: (N, D) numpy array of raw ProtT5 embeddings
        device: Device to run model on
        batch_size: Batch size for projection
        
    Returns:
        (N, D') numpy array of L2-normalized projected embeddings
    """
    model.eval()
    all_projected = []
    
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Projecting embeddings"):
        batch = torch.from_numpy(embeddings[i:i + batch_size]).to(device)
        projected = model(batch)
        # Model already L2-normalizes, but ensure it
        projected = torch.nn.functional.normalize(projected, p=2, dim=1)
        all_projected.append(projected.cpu().numpy())
    
    return np.vstack(all_projected)


def load_cath_embeddings_and_build_index(
    cath_h5_path: Path,
    model: CathSupConModel,
    device: torch.device,
    batch_size: int = 4096,
) -> Tuple[np.ndarray, list[str]]:
    """Load CATH embeddings, project through model, and return projected embeddings + domain IDs.
    
    Args:
        cath_h5_path: Path to CATH H5 embeddings file
        model: Trained model for projection
        device: Device to run model on
        batch_size: Batch size for projection
        
    Returns:
        Tuple of (projected_embeddings, domain_ids) where:
        - projected_embeddings: (N, D) numpy array of L2-normalized projected embeddings
        - domain_ids: List of N domain IDs corresponding to embeddings
    """
    logger.info(f"Loading CATH embeddings from: {cath_h5_path}")
    
    with EmbeddingReader(cath_h5_path) as reader:
        # Get all keys from H5 file
        if reader.is_h5:
            h5_keys = list(reader.h5_file.keys())
        else:
            # For LMDB, collect all keys
            h5_keys = []
            with reader.env.begin(write=False) as txn:
                cursor = txn.cursor()
                for key_bytes, _ in cursor:
                    h5_keys.append(key_bytes.decode("utf-8"))
        
        logger.info(f"Found {len(h5_keys)} CATH embeddings")
        
        # Load embeddings in batches
        embeddings_list = []
        domain_ids = []
        missing = 0
        
        for batch in tqdm(reader.iter_embeddings(keys=h5_keys, batch_size=2048, to_float32=True), 
                         desc="Loading CATH embeddings"):
            for key, emb in batch:
                if emb is None:
                    missing += 1
                    continue
                embeddings_list.append(emb)
                domain_ids.append(extract_domain_id(key))
        
        if missing > 0:
            logger.warning(f"Missing {missing} CATH embeddings")
        
        if not embeddings_list:
            raise ValueError(f"No valid CATH embeddings loaded from {cath_h5_path}")
        
        embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)
        logger.info(f"Loaded {len(embeddings_matrix)} CATH embeddings of dimension {embeddings_matrix.shape[1]}")
    
    # Project through model
    logger.info("Projecting CATH embeddings through model...")
    projected_embeddings = project_embeddings(model, embeddings_matrix, device, batch_size)
    
    return projected_embeddings, domain_ids


def load_cath_superfamily_mapping(
    sf_list_path: Path,
    domain_ids: list[str],
) -> Dict[str, str]:
    """Load CATH superfamily mapping for domain IDs.
    
    Args:
        sf_list_path: Path to cath-domain-sf-list.txt
        domain_ids: List of domain IDs to check coverage for (for logging)
        
    Returns:
        Dictionary mapping domain_id -> superfamily_code (all entries from file)
    """
    logger.info(f"Loading CATH superfamily mappings from: {sf_list_path}")
    
    # Load all mappings from file
    id_to_sf: Dict[str, str] = {}
    with open(sf_list_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            # Handle both tab and space separators
            parts = line.strip().split()
            if len(parts) >= 2:
                domain_id, superfamily = parts[0], parts[1]
                id_to_sf[domain_id] = superfamily
    
    logger.info(f"Loaded {len(id_to_sf)} superfamily mappings from file")
    
    # Check coverage for domain IDs we have embeddings for
    domain_set = set(domain_ids)
    mapped_count = sum(1 for did in domain_ids if did in id_to_sf)
    logger.info(f"Mapped {mapped_count}/{len(domain_ids)} CATH domain IDs to superfamilies")
    
    return id_to_sf


def assign_ted_sequences(
    ted_lmdb_path: Path,
    model: CathSupConModel,
    faiss_index,
    ref_domain_ids: np.ndarray,
    domain_id_to_sf: Dict[str, str],
    device: torch.device,
    output_path: Path,
    batch_size: int = 2048,
) -> None:
    """Assign CATH superfamilies to TED sequences using nearest neighbor search.
    
    Args:
        ted_lmdb_path: Path to TED LMDB embeddings directory
        model: Trained model for projection
        faiss_index: FAISS index built from CATH embeddings
        ref_domain_ids: Array of domain IDs corresponding to FAISS index positions
        domain_id_to_sf: Dictionary mapping domain_id -> superfamily_code
        device: Device to run model on
        output_path: Path to output TSV file
        batch_size: Batch size for processing
    """
    logger.info(f"Processing TED embeddings from: {ted_lmdb_path}")
    
    with EmbeddingReader(ted_lmdb_path) as reader:
        total_entries = len(reader)
        logger.info(f"Found {total_entries} TED embeddings")
        
        # Open output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["query_id", "cath_assignment", "distance"])
            
            processed = 0
            missing = 0
            
            # Iterate over all TED embeddings
            for batch in tqdm(reader.iter_embeddings(keys=None, batch_size=batch_size, to_float32=True),
                             desc="Processing TED embeddings",
                             total=(total_entries + batch_size - 1) // batch_size):
                
                batch_keys = []
                batch_embeddings = []
                
                for key, emb in batch:
                    if emb is None:
                        missing += 1
                        query_id = extract_domain_id(key)
                        writer.writerow([query_id, "missing_embedding", ""])
                        continue
                    
                    batch_keys.append(key)
                    batch_embeddings.append(emb)
                
                if batch_embeddings:
                    # Convert to numpy array
                    embeddings_array = np.vstack(batch_embeddings).astype(np.float32)
                    
                    # Project through model
                    batch_tensor = torch.from_numpy(embeddings_array).to(device)
                    with torch.no_grad():
                        projected = model(batch_tensor)
                        projected = torch.nn.functional.normalize(projected, p=2, dim=1)
                        query_vectors = projected.cpu().numpy().astype(np.float32)
                    
                    # Search FAISS index (k=1 for nearest neighbor)
                    similarities, indices = search_faiss_index(faiss_index, query_vectors, k=1)
                    distances = 1.0 - similarities  # Convert similarity to cosine distance
                    
                    # Write results
                    for i, (key, distance) in enumerate(zip(batch_keys, distances[:, 0])):
                        query_id = extract_domain_id(key)
                        
                        # Get nearest neighbor domain ID and superfamily
                        nn_idx = indices[i, 0]
                        nn_domain_id = ref_domain_ids[nn_idx]
                        cath_assignment = domain_id_to_sf.get(nn_domain_id, "unknown")
                        
                        writer.writerow([query_id, cath_assignment, f"{distance:.6f}"])
                        processed += 1
                
                # Flush periodically
                if processed % 10000 == 0:
                    f.flush()
        
        logger.info(f"Processed {processed} TED embeddings")
        if missing > 0:
            logger.warning(f"Missing {missing} TED embeddings")


def main():
    parser = argparse.ArgumentParser(
        description="Assign CATH superfamilies to TED sequences using nearest neighbor search"
    )
    parser.add_argument(
        "--cath-h5",
        type=Path,
        default=Path("data/cath-domain-seqs-S100.h5"),
        help="Path to CATH H5 embeddings file (default: data/cath-domain-seqs-S100.h5)"
    )
    parser.add_argument(
        "--ted-lmdb",
        type=Path,
        default=Path("data/thits_lmdb"),
        help="Path to TED LMDB embeddings directory (default: data/thits_lmdb)"
    )
    parser.add_argument(
        "--model-ckpt",
        type=Path,
        default=Path("checkpoints/holdouts66.ckpt"),
        help="Path to model checkpoint (default: checkpoints/holdouts66.ckpt)"
    )
    parser.add_argument(
        "--sf-list",
        type=Path,
        default=Path("data/cath-domain-sf-list.txt"),
        help="Path to CATH superfamily list file (default: data/cath-domain-sf-list.txt)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ted_cath_assignments.tsv"),
        help="Path to output TSV file (default: outputs/ted_cath_assignments.tsv)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size for processing (default: 2048)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/mps/cpu). Auto-detected if not specified"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Validate inputs
    required_files = [
        (args.cath_h5, "CATH H5 embeddings"),
        (args.ted_lmdb, "TED LMDB directory"),
        (args.model_ckpt, "Model checkpoint"),
        (args.sf_list, "CATH superfamily list"),
    ]
    for path, name in required_files:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
    
    # Load model
    logger.info(f"Loading model from: {args.model_ckpt}")
    model = CathSupConModel.load_from_checkpoint(str(args.model_ckpt), strict=False)
    model.eval()
    model.to(device)
    
    # Load CATH embeddings and build FAISS index
    cath_embeddings, cath_domain_ids = load_cath_embeddings_and_build_index(
        args.cath_h5, model, device, args.batch_size
    )
    
    logger.info("Building FAISS index...")
    faiss_index = build_faiss_index(cath_embeddings)
    ref_domain_ids = np.array(cath_domain_ids)
    logger.info(f"Built FAISS index with {faiss_index.ntotal} vectors")
    
    # Load CATH superfamily mappings
    domain_id_to_sf = load_cath_superfamily_mapping(args.sf_list, cath_domain_ids)
    
    # Assign TED sequences
    assign_ted_sequences(
        args.ted_lmdb,
        model,
        faiss_index,
        ref_domain_ids,
        domain_id_to_sf,
        device,
        args.output,
        args.batch_size,
    )
    
    logger.info(f"✓ Assignment complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
