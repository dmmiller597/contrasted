#!/usr/bin/env python3
"""
Stream a CATH FASTA, embed with ProstT5, and write per-residue embeddings.

Two passes:
  1) Count labeled sequences and total residues for pre-allocation.
  2) Stream, embed, and write sequentially.

Output directory files:
  - residue_embeddings.npy: float16/float32, shape (total_residues, 1024)
  - residue_offsets.npy: int64, shape (N + 1)
  - labels.npy: int64, shape (N,)
  - ids.txt: one domain ID per line
  - metadata.json: embedding_mode, dims, count, total_residues, dtype, source, model
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from contrasted.data import parse_fasta_header
from contrasted.utils import get_device, load_labels

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def sanitize_sequence(seq: str) -> str:
    return seq.replace("U", "X").replace("Z", "X").replace("O", "X")


def iter_fasta(fasta_path: Path):
    """Yield (domain_id, sequence) from FASTA in a streaming manner."""
    current_id = None
    seq_chunks: list[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, "".join(seq_chunks).replace("-", "")
                try:
                    current_id = parse_fasta_header(line)
                except ValueError as exc:
                    logger.warning("Skipping invalid header: %s", exc)
                    current_id = None
                seq_chunks = []
            elif current_id is not None:
                seq_chunks.append(line)

    if current_id is not None:
        yield current_id, "".join(seq_chunks).replace("-", "")


def count_labeled_sequences_and_residues(
    fasta_path: Path,
    id_to_label_idx: dict[str, int],
) -> tuple[int, int]:
    """Count labeled sequences and total residues."""
    n_seq = 0
    n_res = 0
    for domain_id, seq in iter_fasta(fasta_path):
        if domain_id not in id_to_label_idx:
            continue
        n_seq += 1
        n_res += len(sanitize_sequence(seq))
    return n_seq, n_res


def load_prostt5(
    model_dir: str,
    device: torch.device,
    half: bool,
) -> tuple[T5EncoderModel, T5Tokenizer]:
    logger.info("Loading ProstT5 from: %s", model_dir)
    model = T5EncoderModel.from_pretrained(model_dir).to(device).eval()
    if half:
        model = model.half()
        logger.info("Using half precision for ProstT5")
    tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, tokenizer


def format_sequence(seq: str) -> str:
    return "<AA2fold> " + " ".join(list(sanitize_sequence(seq)))


def process_batch_per_residue(
    batch: list[tuple[str, str, int, int]],
    model: T5EncoderModel,
    tokenizer: T5Tokenizer,
    device: torch.device,
) -> tuple[tuple[str, ...], list[torch.Tensor], tuple[int, ...]]:
    """Return one hidden-state row per residue for each sequence in the batch."""
    domain_ids, formatted_seqs, seq_lens, label_idxs = zip(*batch, strict=False)
    token_encoding = tokenizer.batch_encode_plus(
        list(formatted_seqs),
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(
            token_encoding.input_ids,
            attention_mask=token_encoding.attention_mask,
        )
        hidden = output.last_hidden_state
        attn = token_encoding.attention_mask

    per_seq: list[torch.Tensor] = []
    for i, seq_len in enumerate(seq_lens):
        valid_tokens = int(attn[i].sum().item())
        available_residues = max(valid_tokens - 2, 0)
        n_residues = min(int(seq_len), available_residues)
        rows = hidden[i, 1 : 1 + n_residues]
        per_seq.append(rows)

    return domain_ids, per_seq, label_idxs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed CATH FASTA with ProstT5 and save per-residue embeddings"
    )
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Rostlab/ProstT5")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-residues", type=int, default=4000)
    parser.add_argument("--max-batch", type=int, default=100)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1000,
        help="Batching hint: flush when a sequence exceeds this length",
    )
    parser.add_argument("--flush-every", type=int, default=10000)
    args = parser.parse_args()

    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Label file not found: {args.labels}")

    device = get_device()
    logger.info("Using device: %s", device)

    logger.info("Loading labels from: %s", args.labels)
    id_to_label_idx, idx_to_label = load_labels(args.labels)

    logger.info("Counting labeled sequences and residues in FASTA (first pass)...")
    start_count = time.time()
    total, total_res = count_labeled_sequences_and_residues(
        args.fasta,
        id_to_label_idx,
    )
    logger.info(
        "Found %s labeled sequences and %s residues in %.1fs",
        total,
        total_res,
        time.time() - start_count,
    )
    if total == 0:
        raise ValueError("No labeled sequences found in FASTA.")
    if total_res == 0:
        raise ValueError("No residues found in labeled sequences.")

    args.output.mkdir(parents=True, exist_ok=True)
    residue_path = args.output / "residue_embeddings.npy"
    offsets_path = args.output / "residue_offsets.npy"
    labels_path = args.output / "labels.npy"
    ids_path = args.output / "ids.txt"
    metadata_path = args.output / "metadata.json"

    model, tokenizer = load_prostt5(args.model, device, args.half)
    embedding_dim = int(getattr(model.config, "d_model", 1024))

    residue_mm = np.lib.format.open_memmap(
        residue_path,
        mode="w+",
        dtype=np.float16 if args.half else np.float32,
        shape=(total_res, embedding_dim),
    )
    offsets_mm = np.lib.format.open_memmap(
        offsets_path,
        mode="w+",
        dtype=np.int64,
        shape=(total + 1,),
    )
    labels_mm = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.int64,
        shape=(total,),
    )

    batch: list[tuple[str, str, int, int]] = []
    seq_idx = 0
    res_idx = 0
    last_flush_seq = 0
    progress = tqdm(total=total, desc="Embedding", unit="seq")

    def flush_writes() -> None:
        residue_mm.flush()
        offsets_mm.flush()
        labels_mm.flush()

    def write_batch(
        domain_ids: tuple[str, ...],
        per_seq_tensors: list[torch.Tensor],
        label_idxs: tuple[int, ...],
        ids_file,
    ) -> tuple[int, int]:
        nonlocal seq_idx, res_idx
        for did, rt, label_idx in zip(
            domain_ids, per_seq_tensors, label_idxs, strict=True
        ):
            offsets_mm[seq_idx] = res_idx
            arr = rt.detach().cpu().float().numpy()
            arr = arr.astype(residue_mm.dtype, copy=False)
            n_rows = int(arr.shape[0])
            end_r = res_idx + n_rows
            if end_r > total_res:
                raise RuntimeError(
                    f"Residue write exceeded preallocated size ({end_r} > {total_res})"
                )
            residue_mm[res_idx:end_r] = arr
            labels_mm[seq_idx] = int(label_idx)
            ids_file.write(did + "\n")
            seq_idx += 1
            res_idx = end_r
        progress.update(len(domain_ids))
        return seq_idx, res_idx

    try:
        with open(ids_path, "w") as ids_file:
            for domain_id, seq in iter_fasta(args.fasta):
                label_idx = id_to_label_idx.get(domain_id)
                if label_idx is None:
                    continue

                seq = sanitize_sequence(seq)
                seq_len = len(seq)
                batch.append((domain_id, format_sequence(seq), seq_len, label_idx))

                n_res_batch = sum(s_len for _, _, s_len, _ in batch)
                should_process = (
                    len(batch) >= args.max_batch
                    or n_res_batch >= args.max_residues
                    or seq_len > args.max_seq_len
                )

                if should_process:
                    domain_ids, per_seq_tensors, label_idxs = process_batch_per_residue(
                        batch,
                        model,
                        tokenizer,
                        device,
                    )
                    batch = []
                    write_batch(domain_ids, per_seq_tensors, label_idxs, ids_file)

                    if seq_idx - last_flush_seq >= args.flush_every:
                        flush_writes()
                        last_flush_seq = seq_idx

            if batch:
                domain_ids, per_seq_tensors, label_idxs = process_batch_per_residue(
                    batch,
                    model,
                    tokenizer,
                    device,
                )
                write_batch(domain_ids, per_seq_tensors, label_idxs, ids_file)

    finally:
        progress.close()
        flush_writes()

    offsets_mm[seq_idx] = res_idx
    if seq_idx != total or res_idx != total_res:
        raise RuntimeError(
            f"Wrote seq_idx={seq_idx} (expected {total}), "
            f"res_idx={res_idx} (expected {total_res}). "
            "Ensure FASTA headers match the label file."
        )

    metadata = {
        "embedding_mode": "per_residue",
        "dims": int(embedding_dim),
        "count": int(total),
        "total_residues": int(total_res),
        "dtype": str(residue_mm.dtype),
        "source": str(args.fasta),
        "model": args.model,
        "idx_to_label": {int(k): v for k, v in idx_to_label.items()},
        "residue_embeddings_file": residue_path.name,
        "residue_offsets_file": offsets_path.name,
    }
    metadata_path.write_text(json.dumps(metadata))

    gb = residue_path.stat().st_size / (1024**3)
    logger.info("Saved per-residue embeddings to: %s", args.output)
    logger.info("  - %s sequences, %s residue rows", total, total_res)
    logger.info("  - %s dims", embedding_dim)
    logger.info("  - %.2f GB residue_embeddings.npy", gb)


if __name__ == "__main__":
    main()
