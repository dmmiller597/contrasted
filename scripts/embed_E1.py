#!/usr/bin/env python3
"""
Embed CATH sequences using Profluent E1 and save to .pt file.

Creates a .pt file containing:
    - embeddings: Tensor[N, D] float16 - per-protein embeddings
    - labels: Tensor[N] int64 - superfamily label indices
    - ids: List[str] - domain IDs
    - idx_to_label: Dict[int, str] - label index to superfamily code

Usage:
    python embed_E1.py \
        --model Profluent-Bio/E1-600m \
        --max-batch-tokens 65536

Defaults:
    --fasta data/cath-c123-S100.fasta
    --labels data/cath-domain-sf-list.txt
    --output data/cath-c123-S100.e1.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def add_e1_to_path() -> Path:
    """Ensure the E1 source is importable."""
    e1_root = Path(__file__).resolve().parents[1] / "tools" / "E1" / "src"
    if not e1_root.exists():
        raise FileNotFoundError(
            "E1 source not found. Expected at: "
            f"{e1_root}. Please clone E1 into tools/E1."
        )
    sys.path.insert(0, str(e1_root))
    return e1_root


def parse_fasta_header(header: str) -> str:
    """Extract domain_id from CATH FASTA header.

    Format: >cath|{cath_release}|{domain_id}/{start}-{end}
    Returns: domain_id (e.g., '12e8H01')
    """
    parts = header.strip().lstrip(">").split("|")
    if len(parts) >= 3:
        return parts[2].split("/")[0]
    raise ValueError(f"Invalid FASTA header: {header}")


def read_fasta(fasta_path: Path) -> dict[str, str]:
    """Read FASTA file and return dict of domain_id -> sequence."""
    sequences: dict[str, str] = {}
    current_id: str | None = None

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                try:
                    current_id = parse_fasta_header(line)
                    sequences[current_id] = ""
                except ValueError as exc:
                    logger.warning("Skipping invalid header: %s", exc)
                    current_id = None
            elif current_id is not None:
                # Remove gaps and join multi-line sequences
                sequences[current_id] += line.replace("-", "")

    return sequences


def load_labels(label_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load CATH superfamily labels.

    File format: {domain_id}\t{superfamily_code}
    Returns: (domain_id -> sf_idx, sf_idx -> sf_code)
    """
    id_to_sf_idx: dict[str, int] = {}
    sf_to_idx: dict[str, int] = {}

    with open(label_path) as f:
        for line in f:
            if not line.startswith("#"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    domain_id, superfamily = parts[0], parts[1]
                    if superfamily not in sf_to_idx:
                        sf_to_idx[superfamily] = len(sf_to_idx)
                    id_to_sf_idx[domain_id] = sf_to_idx[superfamily]

    idx_to_sf = {v: k for k, v in sf_to_idx.items()}
    logger.info(
        "Loaded %s domain labels, %s superfamilies",
        len(id_to_sf_idx),
        len(idx_to_sf),
    )
    return id_to_sf_idx, idx_to_sf


def embed_sequences(
    sequences: dict[str, str],
    model_name: str,
    max_batch_tokens: int,
) -> dict[str, torch.Tensor]:
    """Embed sequences using E1 mean-token embeddings."""
    add_e1_to_path()
    from E1 import dist
    from E1.modeling import E1ForMaskedLM
    from E1.predictor import E1Predictor

    device = dist.get_device()
    logger.info("Using device: %s", device)

    logger.info("Loading E1 model: %s", model_name)
    model = E1ForMaskedLM.from_pretrained(model_name, dtype=torch.float).to(device).eval()

    predictor = E1Predictor(
        model=model,
        max_batch_tokens=max_batch_tokens,
        fields_to_save=["mean_token_embeddings"],
        keep_predictions_in_gpu=False,
    )

    sequence_ids = list(sequences.keys())
    sequence_values = list(sequences.values())
    embeddings: dict[str, torch.Tensor] = {}

    start_time = time.time()
    logger.info("Embedding %s sequences", len(sequence_ids))

    for prediction in predictor.predict(sequences=sequence_values, sequence_ids=sequence_ids):
        domain_id = prediction["id"]
        embeddings[domain_id] = prediction["mean_token_embeddings"].cpu()

    elapsed = time.time() - start_time
    if embeddings:
        logger.info(
            "Embedded %s sequences in %.1fs (%.3fs per sequence)",
            len(embeddings),
            elapsed,
            elapsed / len(embeddings),
        )
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed CATH sequences with E1 and save to .pt file"
    )
    parser.add_argument(
        "--fasta",
        "-f",
        type=Path,
        default=Path("data/cath-c123-S100.fasta"),
        help="Path to CATH FASTA file (default: data/cath-c123-S100.fasta)",
    )
    parser.add_argument(
        "--labels",
        "-l",
        type=Path,
        default=Path("data/cath-domain-sf-list.txt"),
        help="Path to cath-domain-sf-list.txt label file (default: data/cath-domain-sf-list.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/cath-c123-S100.e1.pt"),
        help="Output .pt file path (default: data/cath-c123-S100.e1.pt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Profluent-Bio/E1-600m",
        help="HuggingFace model name or local path (default: Profluent-Bio/E1-600m)",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=65536,
        help="Max tokens per batch (default: 65536)",
    )

    args = parser.parse_args()

    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Label file not found: {args.labels}")

    logger.info("Reading FASTA: %s", args.fasta)
    sequences = read_fasta(args.fasta)
    logger.info("Read %s sequences", len(sequences))

    logger.info("Loading labels: %s", args.labels)
    id_to_label_idx, idx_to_label = load_labels(args.labels)

    sequences = {k: v for k, v in sequences.items() if k in id_to_label_idx}
    logger.info("Filtered to %s sequences with labels", len(sequences))
    if not sequences:
        raise ValueError("No labeled sequences found after filtering.")

    embeddings_dict = embed_sequences(
        sequences,
        model_name=args.model,
        max_batch_tokens=args.max_batch_tokens,
    )

    domain_ids = list(embeddings_dict.keys())
    embeddings_list = [embeddings_dict[did] for did in domain_ids]
    labels_list = [id_to_label_idx[did] for did in domain_ids]

    embeddings_tensor = torch.stack(embeddings_list).half()
    labels_tensor = torch.tensor(labels_list, dtype=torch.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings_tensor,
            "labels": labels_tensor,
            "ids": domain_ids,
            "idx_to_label": idx_to_label,
        },
        args.output,
    )

    file_size_mb = args.output.stat().st_size / (1024 * 1024)
    logger.info("Saved to: %s", args.output)
    logger.info("  - %s sequences", len(domain_ids))
    logger.info("  - %s embedding dims", embeddings_tensor.shape[1])
    logger.info("  - %s classes", len(idx_to_label))
    logger.info("  - %.1f MB", file_size_mb)


if __name__ == "__main__":
    main()
