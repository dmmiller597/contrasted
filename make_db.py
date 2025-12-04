"""Create a FAISS vector database from trained model embeddings."""

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import h5py

from contrasted.utils import load_h5_keys_from_fasta, extract_domain_id, normalize_h5_key
from contrasted.model import CathSupConModel
from contrasted.faiss_utils import build_faiss_index

logger = logging.getLogger(__name__)


def load_embeddings_h5(
    h5_path: Path,
    h5_keys: list[str],
    num_workers: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """Load embeddings from HDF5 using parallel reads."""
    
    def read_chunk(chunk_keys):
        results = []
        with h5py.File(h5_path, 'r') as f:
            for key in chunk_keys:
                try:
                    normalized = normalize_h5_key(key)
                    emb = np.array(f[normalized])
                    results.append((key, emb))
                except KeyError:
                    results.append((key, None))
        return results
    
    chunk_size = max(1, len(h5_keys) // num_workers)
    chunks = [h5_keys[i:i + chunk_size] for i in range(0, len(h5_keys), chunk_size)]
    
    all_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(read_chunk, chunk) for chunk in chunks]
        for future in tqdm(futures, desc="Loading"):
            all_results.extend(future.result())
    
    embeddings = []
    domain_ids = []
    missing = 0
    for key, emb in all_results:
        if emb is None:
            missing += 1
            continue
        embeddings.append(emb)
        domain_ids.append(extract_domain_id(key))
    
    if missing > 0:
        logger.warning(f"Missing {missing} embeddings")
    
    return np.vstack(embeddings).astype(np.float32), domain_ids


@torch.no_grad()
def project_embeddings(
    model: CathSupConModel,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Project embeddings through trained model."""
    model.eval()
    all_projected = []
    
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Projecting"):
        batch = torch.from_numpy(embeddings[i:i + batch_size]).to(device)
        projected = model(batch)
        all_projected.append(projected.cpu().numpy())
    
    return np.vstack(all_projected)


@hydra.main(version_base=None, config_path="configs", config_name="make_db")
def main(cfg: DictConfig):
    """Create FAISS index from trained model embeddings."""
    
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
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = CathSupConModel.load_from_checkpoint(str(model_path), strict=False)
    model.eval()
    model.to(device)
    
    # Load FASTA and get HDF5 keys
    logger.info(f"Loading sequences from: {input_path}")
    h5_keys = load_h5_keys_from_fasta(input_path)
    
    # Filter by IDs if specified
    if cfg.ids:
        ids_set = set(cfg.ids)
        h5_keys = [k for k in h5_keys if extract_domain_id(k) in ids_set]
        logger.info(f"Filtered to {len(h5_keys)} sequences matching provided IDs")
    else:
        logger.info(f"Processing {len(h5_keys)} sequences")
    
    # Load embeddings and project through model
    embedding_file = Path(cfg.get("embedding_file", "data/cath-domain-seqs-S100.h5"))
    logger.info(f"Loading embeddings from: {embedding_file}")
    
    raw_embeddings, domain_ids = load_embeddings_h5(embedding_file, h5_keys)
    embeddings_matrix = project_embeddings(model, raw_embeddings, device)
    
    # Validate embeddings
    if embeddings_matrix.size == 0:
        raise ValueError("No valid embeddings generated")
    
    logger.info(
        f"Generated {embeddings_matrix.shape[0]} embeddings "
        f"of dimension {embeddings_matrix.shape[1]}"
    )
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    index = build_faiss_index(embeddings_matrix)
    
    # Save index
    index_path = Path(cfg.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    import faiss
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to: {index_path}")
    
    # Save domain IDs
    ids_path = index_path.with_suffix(".npy")
    np.save(ids_path, np.array(domain_ids))
    logger.info(f"Saved domain IDs to: {ids_path}")
    
    logger.info("✓ Database creation complete!")


if __name__ == "__main__":
    main()
