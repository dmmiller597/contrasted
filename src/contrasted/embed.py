"""Generate ProstT5 protein embeddings directly from FASTA.

The public API is intentionally small:

- :class:`EncodeConfig` -- tunable batching / precision knobs.
- :class:`ProstT5Encoder` -- loads the model lazily; ``encode`` a dict of
  ``{id: sequence}`` -> ``{id: (1024,) ndarray}``.
- :func:`encode_fasta` -- FASTA path(s) -> :class:`EmbeddingStore`.
- :func:`encode_fasta_to_dir` -- FASTA path(s) -> persisted embedding dir.
- :func:`main` -- Hydra entrypoint for ``contrasted-embed``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from contrasted.data import EmbeddingStore, read_fasta_sequences
from contrasted.utils import get_device, load_labels

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

PROSTT5_MODEL = "Rostlab/ProstT5"
PROSTT5_PREFIX = "<AA2fold>"
PROSTT5_DIM = 1024

_NON_STANDARD_AA = str.maketrans({"U": "X", "Z": "X", "O": "X"})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EncodeConfig:
    """Tunable knobs for ProstT5 encoding."""

    model_name: str = PROSTT5_MODEL
    half: bool = True  # forced off on CPU/MPS (unsafe / slow)
    max_residues: int = 4000
    max_batch: int = 100
    max_seq_len: int = 1000
    dtype: str = "float16"  # storage dtype for saved embeddings
    device: torch.device | None = None


# ---------------------------------------------------------------------------
# Pure helpers (model-free; covered by unit tests)
# ---------------------------------------------------------------------------


def _format_sequence(seq: str) -> str:
    """ProstT5 input format: ``"<AA2fold> R E S I D U E S"`` with non-standard
    amino acids (U/Z/O) masked as X."""
    return PROSTT5_PREFIX + " " + " ".join(seq.translate(_NON_STANDARD_AA))


def _build_batches(
    items: Sequence[tuple[str, str]],
    *,
    max_residues: int,
    max_batch: int,
    max_seq_len: int,
) -> list[list[tuple[str, str, int]]]:
    """Group ``(domain_id, sequence)`` pairs into batches for the encoder.

    A batch flushes when it hits ``max_batch`` items, accumulates
    ``max_residues`` total residues, or contains a sequence longer than
    ``max_seq_len`` (processed alone for OOM safety).
    """
    batches: list[list[tuple[str, str, int]]] = []
    current: list[tuple[str, str, int]] = []
    residues = 0
    for domain_id, seq in items:
        seq_len = len(seq)
        current.append((domain_id, _format_sequence(seq), seq_len))
        residues += seq_len
        if (
            len(current) >= max_batch
            or residues >= max_residues
            or seq_len > max_seq_len
        ):
            batches.append(current)
            current = []
            residues = 0
    if current:
        batches.append(current)
    return batches


def _resolve_device(
    cfg: EncodeConfig,
) -> tuple[torch.device, bool]:
    """Return ``(device, effective_half)`` with safe overrides."""
    device = cfg.device if cfg.device is not None else get_device()
    half = cfg.half and device.type == "cuda"
    if cfg.half and device.type != "cuda":
        logger.info(
            f"Disabling half precision on device={device.type}; full precision only."
        )
    return device, half


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class ProstT5Encoder:
    """Lazily-loaded ProstT5 wrapper. Holds the model across calls.

    Usage::

        with ProstT5Encoder() as enc:
            embeddings = enc.encode({"id1": "MSE...", "id2": "ALG..."})
    """

    def __init__(self, config: EncodeConfig | None = None) -> None:
        self.config = config or EncodeConfig()
        self._device, self._half = _resolve_device(self.config)
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import T5EncoderModel, T5Tokenizer  # lazy

        logger.info(f"Loading ProstT5 from: {self.config.model_name}")
        model = T5EncoderModel.from_pretrained(self.config.model_name).to(self._device)
        model.eval()
        if self._half:
            model = model.half()
            logger.info("Using half precision")
        self._model = model
        self._tokenizer = T5Tokenizer.from_pretrained(
            self.config.model_name, do_lower_case=False
        )

    def close(self) -> None:
        """Release the underlying model (frees GPU memory)."""
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> ProstT5Encoder:
        self._ensure_loaded()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @torch.inference_mode()
    def encode(self, sequences: Mapping[str, str]) -> dict[str, np.ndarray]:
        """Encode ``{id: sequence}`` -> ``{id: (1024,) ndarray}``.

        Mean-pools residue-token hidden states (excluding the ``<AA2fold>``
        prefix). Output dtype follows ``config.dtype``. Sequences are
        processed in length-sorted batches for OOM safety; output order
        matches the input mapping.
        """
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        items = list(sequences.items())
        if not items:
            return {}

        # Sort desc by length so OOM trips on the first (largest) batch.
        sorted_items = sorted(items, key=lambda kv: len(kv[1]), reverse=True)
        batches = _build_batches(
            sorted_items,
            max_residues=self.config.max_residues,
            max_batch=self.config.max_batch,
            max_seq_len=self.config.max_seq_len,
        )

        np_dtype = np.dtype(self.config.dtype)
        out: dict[str, np.ndarray] = {}
        start = time.time()
        total_residues = sum(len(s) for s in sequences.values())
        logger.info(
            f"Encoding {len(items)} sequences ({total_residues} residues) "
            f"in {len(batches)} batches on {self._device}"
        )

        for batch in tqdm(batches, desc="Encoding", leave=False):
            ids, formatted, seq_lens = zip(*batch, strict=False)
            encoded = self._tokenizer.batch_encode_plus(
                list(formatted),
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            ).to(self._device)

            try:
                output = self._model(
                    encoded.input_ids, attention_mask=encoded.attention_mask
                )
            except RuntimeError as e:
                logger.error(f"Encoder failed on batch starting {ids[0]}: {e}")
                continue

            hidden = output.last_hidden_state
            for i, did in enumerate(ids):
                s_len = seq_lens[i]
                # +1 skips the <AA2fold> prefix token.
                emb = hidden[i, 1 : s_len + 1].mean(dim=0)
                out[did] = emb.to(torch.float32).cpu().numpy().astype(np_dtype)

        elapsed = time.time() - start
        if out:
            logger.info(
                f"Encoded {len(out)} sequences in {elapsed:.1f}s "
                f"({elapsed / len(out) * 1000:.1f} ms/seq)"
            )
        # Reorder to match input.
        return {did: out[did] for did in sequences if did in out}


# ---------------------------------------------------------------------------
# Public convenience API
# ---------------------------------------------------------------------------


def _normalize_fasta_paths(
    fasta: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(fasta, (str, Path)):
        return [Path(fasta)]
    return [Path(p) for p in fasta]


def _collect_sequences(fasta_paths: list[Path]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    for path in fasta_paths:
        if not path.exists():
            raise FileNotFoundError(f"FASTA not found: {path}")
        for did, seq in read_fasta_sequences(path).items():
            if did in sequences:
                logger.warning(f"Duplicate id '{did}' across FASTAs; keeping first")
                continue
            sequences[did] = seq
    return sequences


def encode_fasta(
    fasta: str | Path | Sequence[str | Path],
    *,
    encoder: ProstT5Encoder | None = None,
    config: EncodeConfig | None = None,
    labels_path: str | Path | None = None,
) -> EmbeddingStore:
    """Encode one or more FASTA files and return an :class:`EmbeddingStore`.

    If ``labels_path`` is provided, rows without a matching label are
    dropped with a warning. The caller owns the returned encoder if they
    pass one in; otherwise a temporary encoder is created and closed.
    """
    fasta_paths = _normalize_fasta_paths(fasta)
    sequences = _collect_sequences(fasta_paths)
    if not sequences:
        raise ValueError(f"No sequences parsed from: {fasta_paths}")

    owned = encoder is None
    enc = encoder or ProstT5Encoder(config or EncodeConfig())
    try:
        vectors = enc.encode(sequences)
    finally:
        if owned:
            enc.close()

    if not vectors:
        raise RuntimeError("Encoder returned no embeddings (all batches failed)")

    id_to_label_idx: dict[str, int] | None = None
    idx_to_label: dict[int, str] | None = None
    if labels_path is not None:
        id_to_label_idx, idx_to_label = load_labels(labels_path)
        missing = [did for did in vectors if did not in id_to_label_idx]
        if missing:
            logger.warning(
                f"{len(missing)}/{len(vectors)} sequences lack a label and will "
                f"be dropped. First missing: {missing[:3]}"
            )
        vectors = {did: v for did, v in vectors.items() if did in id_to_label_idx}
        if not vectors:
            raise ValueError("No sequences remain after label filtering")

    ids = list(vectors.keys())
    embeddings = np.stack([vectors[did] for did in ids], axis=0)
    labels_arr: np.ndarray | None = None
    if id_to_label_idx is not None:
        labels_arr = np.asarray([id_to_label_idx[did] for did in ids], dtype=np.int64)

    return EmbeddingStore(
        embeddings=embeddings,
        ids=ids,
        labels=labels_arr,
        id_to_idx={did: i for i, did in enumerate(ids)},
        idx_to_label=idx_to_label,
    )


def encode_fasta_to_dir(
    fasta: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    *,
    encoder: ProstT5Encoder | None = None,
    config: EncodeConfig | None = None,
    labels_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Encode FASTA(s) and persist the 4-file embedding directory.

    If ``overwrite`` is True and ``output_dir`` already contains the
    canonical files, they are removed first. Otherwise a pre-existing
    store raises :class:`FileExistsError`.
    """
    output_dir = Path(output_dir)
    if overwrite and output_dir.is_dir():
        for name in (
            "embeddings.npy",
            "labels.npy",
            "ids.txt",
            "id_to_row.npy",
            "metadata.json",
        ):
            (output_dir / name).unlink(missing_ok=True)

    store = encode_fasta(fasta, encoder=encoder, config=config, labels_path=labels_path)
    source = ",".join(str(p) for p in _normalize_fasta_paths(fasta))
    return store.save(output_dir, source=source)


# ---------------------------------------------------------------------------
# Hydra entrypoint: contrasted-embed
# ---------------------------------------------------------------------------


def build_encode_config(cfg: DictConfig | None) -> EncodeConfig:
    """Translate a Hydra ``embed:`` block (or ``None``) into an ``EncodeConfig``."""
    if cfg is None:
        return EncodeConfig()
    device: torch.device | None = None
    dev_str = cfg.get("device")
    if dev_str:
        device = torch.device(dev_str)
    return EncodeConfig(
        model_name=str(cfg.get("model_name", PROSTT5_MODEL)),
        half=bool(cfg.get("half", True)),
        max_residues=int(cfg.get("max_residues", 4000)),
        max_batch=int(cfg.get("max_batch", 100)),
        max_seq_len=int(cfg.get("max_seq_len", 1000)),
        dtype=str(cfg.get("dtype", "float16")),
        device=device,
    )


def run(cfg: DictConfig) -> None:
    """Encode FASTA(s) to an embedding directory (Hydra config)."""
    input_path = Path(cfg.input)
    output_dir = Path(cfg.output_dir)
    labels_path = Path(cfg.labels) if cfg.get("labels") else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if labels_path is not None and not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Accept a single FASTA or a directory of FASTAs.
    if input_path.is_dir():
        fasta_paths: list[Path] = sorted(input_path.glob("*.fasta")) + sorted(
            input_path.glob("*.fa")
        )
        if not fasta_paths:
            raise FileNotFoundError(f"No FASTA files in: {input_path}")
    else:
        fasta_paths = [input_path]

    out = encode_fasta_to_dir(
        fasta_paths,
        output_dir,
        config=build_encode_config(cfg),
        labels_path=labels_path,
        overwrite=bool(cfg.get("overwrite", False)),
    )
    logger.info(f"Wrote embedding directory: {out}")


@hydra.main(version_base=None, config_path="../../configs", config_name="embed")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI wrapper
    run(cfg)


if __name__ == "__main__":
    main()
