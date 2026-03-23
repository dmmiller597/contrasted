#!/usr/bin/env python3
"""
Stream a CATH FASTA, embed with ESM-2 8M (HuggingFace), write embedding directory.

Two passes:
  1) Count labeled sequences to pre-allocate memmapped outputs.
  2) Stream, embed, and write sequentially.

Output — sequence-level (default, compatible with contrasted.data.load_embedding_dir):
  - embeddings.npy: float16/float32, shape (N, hidden_size)
  - labels.npy: int64, shape (N,)
  - ids.txt: one domain ID per line
  - metadata.json: dims, count, dtype, source, idx_to_label

Output — --per-residue (not load_embedding_dir-compatible):
  - residue_embeddings.npy: float16/float32, shape (total_residues, hidden_size)
  - residue_offsets.npy: int64, shape (N + 1); sequence i is
    residue_embeddings[offsets[i] : offsets[i + 1]]
  - labels.npy, ids.txt, metadata.json (count = N sequences; total_residues in metadata)

Usage:
    uv sync --extra embed
    uv run python scripts/embed_esm2_8m.py \\
        --fasta data/cath-sequence-data/cath-domain-seqs-S100-c123.fasta \\
        --labels data/cath-domain-sf-list.txt \\
        --output data/cath-s100-c123-esm2-8m \\
        --half
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, EsmModel

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


def count_labeled_sequences(
    fasta_path: Path, id_to_label_idx: dict[str, int]
) -> int:
    """Count FASTA entries that have labels."""
    total = 0
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            try:
                domain_id = parse_fasta_header(line)
            except ValueError:
                continue
            if domain_id in id_to_label_idx:
                total += 1
    return total


def count_labeled_sequences_and_residues(
    fasta_path: Path,
    id_to_label_idx: dict[str, int],
    max_residues_per_seq: int,
) -> tuple[int, int]:
    """Count labeled sequences and total stored residue rows (after truncation cap)."""
    n_seq = 0
    n_res = 0
    for domain_id, seq in iter_fasta(fasta_path):
        if domain_id not in id_to_label_idx:
            continue
        seq = sanitize_sequence(seq)
        n_seq += 1
        n_res += min(len(seq), max_residues_per_seq)
    return n_seq, n_res


def resolve_max_token_length(tokenizer: Any, config: Any) -> int:
    cfg_max = int(getattr(config, "max_position_embeddings", 1026))
    tok_ml = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_ml, int) and tok_ml < 1_000_000:
        return min(cfg_max, tok_ml)
    return cfg_max


def load_esm(
    model_dir: str,
    device: torch.device,
    half: bool,
    tokenizer: Any | None = None,
) -> tuple[EsmModel, Any, int]:
    logger.info("Loading ESM from: %s", model_dir)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = EsmModel.from_pretrained(
        model_dir, add_pooling_layer=False
    ).to(device).eval()
    if half:
        model = model.half()
        logger.info("Using half precision for ESM")
    embedding_dim = int(model.config.hidden_size)
    return model, tokenizer, embedding_dim


def mean_pool_hidden(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean pool over non-padding positions (attention_mask: 1 = keep)."""
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).to(
        dtype=last_hidden_state.dtype
    )
    return summed / counts


def process_batch(
    batch: list[tuple[str, str, int, int]],
    model: EsmModel,
    tokenizer: Any,
    device: torch.device,
    max_token_length: int,
) -> tuple[tuple[str, ...], torch.Tensor, tuple[int, ...]]:
    domain_ids, seqs, _seq_lens, label_idxs = zip(*batch, strict=False)

    encoded = tokenizer(
        list(seqs),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_token_length,
    ).to(device)

    with torch.no_grad():
        out = model(**encoded)
        pooled = mean_pool_hidden(out.last_hidden_state, encoded["attention_mask"])

    return domain_ids, pooled, label_idxs


def process_batch_per_residue(
    batch: list[tuple[str, str, int, int]],
    model: EsmModel,
    tokenizer: Any,
    device: torch.device,
    max_token_length: int,
) -> tuple[tuple[str, ...], list[torch.Tensor], tuple[int, ...]]:
    """One hidden vector per amino acid (CLS/EOS dropped).

    Row count matches the truncated sequence length in residues.
    """
    domain_ids, seqs, _seq_lens, label_idxs = zip(*batch, strict=False)

    encoded = tokenizer(
        list(seqs),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_token_length,
    ).to(device)

    with torch.no_grad():
        out = model(**encoded)
        hidden = out.last_hidden_state
        attn = encoded["attention_mask"]

    per_seq: list[torch.Tensor] = []
    for i in range(hidden.shape[0]):
        mask = attn[i].bool()
        row = hidden[i][mask]
        if row.shape[0] >= 2:
            row = row[1:-1]
        per_seq.append(row)

    return domain_ids, per_seq, label_idxs


def max_residues_stored_per_sequence(max_token_length: int) -> int:
    """HF ESM-2 uses CLS + residues + EOS inside max_token_length."""
    return max(0, int(max_token_length) - 2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed CATH FASTA with ESM-2 8M and save embedding directory"
    )
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace model id (default: ESM-2 8M)",
    )
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-residues", type=int, default=4000)
    parser.add_argument("--max-batch", type=int, default=100)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1000,
        help="Batching: flush when a sequence exceeds this length (residues)",
    )
    parser.add_argument("--flush-every", type=int, default=10000)
    parser.add_argument(
        "--per-residue",
        action="store_true",
        help=(
            "Per-amino-acid rows: residue_embeddings.npy + residue_offsets.npy "
            "(not load_embedding_dir-compatible)"
        ),
    )
    args = parser.parse_args()

    if not args.fasta.exists():
        raise FileNotFoundError(f"FASTA file not found: {args.fasta}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Label file not found: {args.labels}")

    device = get_device()
    logger.info("Using device: %s", device)

    logger.info("Loading labels from: %s", args.labels)
    id_to_label_idx, idx_to_label = load_labels(args.labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_config = AutoConfig.from_pretrained(args.model)
    max_token_length = resolve_max_token_length(tokenizer, hf_config)
    max_aa = max_residues_stored_per_sequence(max_token_length)
    hidden_guess = int(getattr(hf_config, "hidden_size", 320))
    logger.info(
        "Max tokens %s → up to %s residue rows stored per sequence (truncation cap)",
        max_token_length,
        max_aa,
    )

    logger.info("Counting labeled entries in FASTA (first pass)...")
    start_count = time.time()
    if args.per_residue:
        total, total_res = count_labeled_sequences_and_residues(
            args.fasta, id_to_label_idx, max_aa
        )
        bytes_per = 2 if args.half else 4
        approx_gb = total_res * hidden_guess * bytes_per / (1024**3)
        logger.info(
            "Found %s sequences, %s residue rows in %.1fs "
            "(~%.2f GB for residue_embeddings.npy, %s, dim=%s)",
            total,
            total_res,
            time.time() - start_count,
            approx_gb,
            "float16" if args.half else "float32",
            hidden_guess,
        )
    else:
        total = count_labeled_sequences(args.fasta, id_to_label_idx)
        total_res = 0
        logger.info(
            "Found %s labeled sequences in %.1fs",
            total,
            time.time() - start_count,
        )

    if total == 0:
        raise ValueError("No labeled sequences found in FASTA.")
    if args.per_residue and total_res == 0:
        raise ValueError("No residues to store (unexpected).")

    args.output.mkdir(parents=True, exist_ok=True)
    labels_path = args.output / "labels.npy"
    ids_path = args.output / "ids.txt"
    metadata_path = args.output / "metadata.json"

    model, tokenizer, embedding_dim = load_esm(
        args.model, device, args.half, tokenizer=tokenizer
    )

    idx_to_label_json = {int(k): v for k, v in idx_to_label.items()}

    if args.per_residue:
        _run_per_residue_embedding_pass(
            args=args,
            id_to_label_idx=id_to_label_idx,
            idx_to_label_json=idx_to_label_json,
            total=total,
            total_res=total_res,
            max_token_length=max_token_length,
            max_aa=max_aa,
            model=model,
            tokenizer=tokenizer,
            device=device,
            embedding_dim=embedding_dim,
            labels_path=labels_path,
            ids_path=ids_path,
            metadata_path=metadata_path,
        )
    else:
        _run_sequence_embedding_pass(
            args=args,
            id_to_label_idx=id_to_label_idx,
            idx_to_label_json=idx_to_label_json,
            total=total,
            max_token_length=max_token_length,
            model=model,
            tokenizer=tokenizer,
            device=device,
            embedding_dim=embedding_dim,
            labels_path=labels_path,
            ids_path=ids_path,
            metadata_path=metadata_path,
        )


def _run_sequence_embedding_pass(
    *,
    args: argparse.Namespace,
    id_to_label_idx: dict[str, int],
    idx_to_label_json: dict[int, str],
    total: int,
    max_token_length: int,
    model: EsmModel,
    tokenizer: Any,
    device: torch.device,
    embedding_dim: int,
    labels_path: Path,
    ids_path: Path,
    metadata_path: Path,
) -> None:
    embeddings_path = args.output / "embeddings.npy"
    embeddings_mm = np.lib.format.open_memmap(
        embeddings_path,
        mode="w+",
        dtype=np.float16 if args.half else np.float32,
        shape=(total, embedding_dim),
    )
    labels_mm = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int64, shape=(total,)
    )

    batch: list[tuple[str, str, int, int]] = []
    write_idx = 0
    last_flush = 0
    progress = tqdm(total=total, desc="Embedding", unit="seq")

    try:
        with open(ids_path, "w") as ids_file:
            for domain_id, seq in iter_fasta(args.fasta):
                label_idx = id_to_label_idx.get(domain_id)
                if label_idx is None:
                    continue

                seq = sanitize_sequence(seq)
                seq_len = len(seq)
                batch.append((domain_id, seq, seq_len, label_idx))

                n_res_batch = sum(s_len for _, _, s_len, _ in batch)
                should_process = (
                    len(batch) >= args.max_batch
                    or n_res_batch >= args.max_residues
                    or seq_len > args.max_seq_len
                )

                if should_process:
                    domain_ids, pooled, label_idxs = process_batch(
                        batch, model, tokenizer, device, max_token_length
                    )
                    batch = []

                    pooled_np = pooled.detach().cpu().float().numpy()
                    if args.half:
                        pooled_np = pooled_np.astype(np.float16, copy=False)
                    else:
                        pooled_np = pooled_np.astype(np.float32, copy=False)

                    end_idx = write_idx + pooled_np.shape[0]
                    if end_idx > total:
                        raise RuntimeError(
                            "Write index exceeded preallocated size "
                            f"({end_idx} > {total})"
                        )
                    embeddings_mm[write_idx:end_idx] = pooled_np
                    labels_mm[write_idx:end_idx] = np.asarray(
                        label_idxs, dtype=np.int64
                    )
                    ids_file.write("\n".join(domain_ids) + "\n")

                    write_idx = end_idx
                    progress.update(len(domain_ids))

                    if write_idx - last_flush >= args.flush_every:
                        embeddings_mm.flush()
                        labels_mm.flush()
                        last_flush = write_idx

            if batch:
                domain_ids, pooled, label_idxs = process_batch(
                    batch, model, tokenizer, device, max_token_length
                )
                pooled_np = pooled.detach().cpu().float().numpy()
                if args.half:
                    pooled_np = pooled_np.astype(np.float16, copy=False)
                else:
                    pooled_np = pooled_np.astype(np.float32, copy=False)

                end_idx = write_idx + pooled_np.shape[0]
                if end_idx > total:
                    raise RuntimeError(
                        "Write index exceeded preallocated size "
                        f"({end_idx} > {total})"
                    )
                embeddings_mm[write_idx:end_idx] = pooled_np
                labels_mm[write_idx:end_idx] = np.asarray(label_idxs, dtype=np.int64)
                ids_file.write("\n".join(domain_ids) + "\n")
                write_idx = end_idx
                progress.update(len(domain_ids))

    finally:
        progress.close()
        embeddings_mm.flush()
        labels_mm.flush()

    if write_idx != total:
        raise RuntimeError(
            f"Wrote {write_idx} embeddings but expected {total}. "
            "Ensure FASTA headers match the label file."
        )

    metadata = {
        "embedding_mode": "per_sequence",
        "dims": int(embeddings_mm.shape[1]),
        "count": int(total),
        "dtype": str(embeddings_mm.dtype),
        "source": str(args.fasta),
        "model": args.model,
        "idx_to_label": idx_to_label_json,
    }
    metadata_path.write_text(json.dumps(metadata))

    file_size_gb = embeddings_path.stat().st_size / (1024**3)
    logger.info("Saved embeddings to: %s", args.output)
    logger.info("  - %s sequences", total)
    logger.info("  - %s dims", embeddings_mm.shape[1])
    logger.info("  - %.2f GB embeddings.npy", file_size_gb)


def _run_per_residue_embedding_pass(
    *,
    args: argparse.Namespace,
    id_to_label_idx: dict[str, int],
    idx_to_label_json: dict[int, str],
    total: int,
    total_res: int,
    max_token_length: int,
    max_aa: int,
    model: EsmModel,
    tokenizer: Any,
    device: torch.device,
    embedding_dim: int,
    labels_path: Path,
    ids_path: Path,
    metadata_path: Path,
) -> None:
    residue_path = args.output / "residue_embeddings.npy"
    offsets_path = args.output / "residue_offsets.npy"

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
        labels_path, mode="w+", dtype=np.int64, shape=(total,)
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

    try:
        with open(ids_path, "w") as ids_file:
            for domain_id, seq in iter_fasta(args.fasta):
                label_idx = id_to_label_idx.get(domain_id)
                if label_idx is None:
                    continue

                seq = sanitize_sequence(seq)
                seq_len = len(seq)
                batch.append((domain_id, seq, seq_len, label_idx))

                n_res_batch = sum(s_len for _, _, s_len, _ in batch)
                should_process = (
                    len(batch) >= args.max_batch
                    or n_res_batch >= args.max_residues
                    or seq_len > args.max_seq_len
                )

                if should_process:
                    domain_ids, per_seq_tensors, label_idxs = (
                        process_batch_per_residue(
                            batch,
                            model,
                            tokenizer,
                            device,
                            max_token_length,
                        )
                    )
                    batch = []

                    for did, rt, li in zip(
                        domain_ids, per_seq_tensors, label_idxs, strict=True
                    ):
                        offsets_mm[seq_idx] = res_idx
                        arr = rt.detach().cpu().float().numpy()
                        if args.half:
                            arr = arr.astype(np.float16, copy=False)
                        else:
                            arr = arr.astype(np.float32, copy=False)
                        L = int(arr.shape[0])
                        end_r = res_idx + L
                        if end_r > total_res:
                            raise RuntimeError(
                                "Residue write exceeded preallocated size "
                                f"({end_r} > {total_res})"
                            )
                        residue_mm[res_idx:end_r] = arr
                        labels_mm[seq_idx] = int(li)
                        ids_file.write(did + "\n")
                        seq_idx += 1
                        res_idx = end_r

                    progress.update(len(domain_ids))

                    if seq_idx - last_flush_seq >= args.flush_every:
                        flush_writes()
                        last_flush_seq = seq_idx

            if batch:
                domain_ids, per_seq_tensors, label_idxs = (
                    process_batch_per_residue(
                        batch,
                        model,
                        tokenizer,
                        device,
                        max_token_length,
                    )
                )
                for did, rt, li in zip(
                    domain_ids, per_seq_tensors, label_idxs, strict=True
                ):
                    offsets_mm[seq_idx] = res_idx
                    arr = rt.detach().cpu().float().numpy()
                    if args.half:
                        arr = arr.astype(np.float16, copy=False)
                    else:
                        arr = arr.astype(np.float32, copy=False)
                    L = int(arr.shape[0])
                    end_r = res_idx + L
                    if end_r > total_res:
                        raise RuntimeError(
                            "Residue write exceeded preallocated size "
                            f"({end_r} > {total_res})"
                        )
                    residue_mm[res_idx:end_r] = arr
                    labels_mm[seq_idx] = int(li)
                    ids_file.write(did + "\n")
                    seq_idx += 1
                    res_idx = end_r

                progress.update(len(domain_ids))

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
        "idx_to_label": idx_to_label_json,
        "residue_embeddings_file": "residue_embeddings.npy",
        "residue_offsets_file": "residue_offsets.npy",
        "max_residues_per_sequence": int(max_aa),
    }
    metadata_path.write_text(json.dumps(metadata))

    gb = residue_path.stat().st_size / (1024**3)
    logger.info("Saved per-residue embeddings to: %s", args.output)
    logger.info("  - %s sequences, %s residue rows", total, total_res)
    logger.info("  - %s dims", embedding_dim)
    logger.info("  - %.2f GB residue_embeddings.npy", gb)


if __name__ == "__main__":
    main()
