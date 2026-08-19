"""Build AA∥3Di concatenated ProstT5 embedding stores.

Headline ContrasTED uses 2048-d inputs formed by concatenating 1024-d AA and
1024-d 3Di ProstT5 embeddings for the same domain id.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contrasted.data import (
    CANONICAL_EMBEDDING_FILES,
    EmbeddingStore,
    read_fasta_sequences,
)
from contrasted.embed import EncodeConfig, ProstT5Encoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConcatStoreRequest:
    """Inputs for building a 2048-d AA∥3Di ``EmbeddingStore``."""

    aa_store: Path
    output_dir: Path
    domain_ids: list[str]
    di_embeddings: np.ndarray | None = None
    di_ids: list[str] | None = None
    di_sequences: dict[str, str] | None = None
    pdb_dir: Path | None = None
    foldseek: Path | None = None
    foldseek_threads: int = 16
    foldseek_work_dir: Path | None = None
    encode_config: EncodeConfig | None = None
    overwrite: bool = False
    source: str = "aa||3di concat"


def normalize_foldseek_3di_id(
    header: str,
    known_ids: Collection[str] | None = None,
) -> str:
    """Normalize a Foldseek FASTA header to its source domain ID."""
    without_chain = header.rsplit("_", 1)[0] if "_" in header else header
    without_pdb = header.removesuffix(".pdb")
    without_pdb_then_chain = (
        without_pdb.rsplit("_", 1)[0] if "_" in without_pdb else without_pdb
    )
    candidates = list(
        dict.fromkeys(
            [
                header,
                without_chain,
                header.removesuffix(".pdb"),
                without_chain.removesuffix(".pdb"),
                without_pdb_then_chain,
            ]
        )
    )

    if known_ids is not None:
        for candidate in candidates:
            if candidate in known_ids:
                return candidate

    return without_chain.removesuffix(".pdb")


def resolve_foldseek(explicit: Path | None = None) -> Path:
    """Locate a Foldseek binary."""
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    found = shutil.which("foldseek")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "foldseek not found on PATH; pass --foldseek /path/to/foldseek"
    )


def extract_3di_from_pdbs(
    *,
    foldseek: Path,
    pdb_dir: Path,
    domain_ids: list[str],
    work_dir: Path,
    threads: int,
) -> dict[str, str]:
    """Run Foldseek ``createdb`` / ``convert2fasta`` for missing 3Di strings."""
    if not domain_ids:
        return {}
    work_dir.mkdir(parents=True, exist_ok=True)
    link_dir = work_dir / "pdb_links"
    link_dir.mkdir(exist_ok=True)
    absent: list[str] = []
    for did in domain_ids:
        link = link_dir / f"{did}.pdb"
        if link.exists() or link.is_symlink():
            continue
        src = pdb_dir / f"{did}.pdb"
        if not src.is_file():
            absent.append(did)
            continue
        link.symlink_to(src.resolve())
    if absent:
        logger.warning("Missing PDBs for %d domains: %s", len(absent), absent[:5])

    db = work_dir / "fsdb"
    subprocess.run(
        [str(foldseek), "createdb", str(link_dir), str(db), "--threads", str(threads)],
        check=True,
    )
    ss = Path(str(db) + "_ss")
    subprocess.run(
        [str(foldseek), "lndb", str(db) + "_h", str(db) + "_ss_h"],
        check=True,
    )
    raw_fa = work_dir / "3di_raw.fasta"
    subprocess.run([str(foldseek), "convert2fasta", str(ss), str(raw_fa)], check=True)

    known_ids = set(domain_ids)
    out: dict[str, str] = {}
    for header, seq in read_fasta_sequences(raw_fa).items():
        domain_id = normalize_foldseek_3di_id(header, known_ids)
        out[domain_id] = seq.lower()
    return out


def concat_aa_di_rows(
    aa_rows: np.ndarray,
    di_rows: np.ndarray,
) -> np.ndarray:
    """Concatenate AA and 3Di rows along the feature axis."""
    if aa_rows.shape[0] != di_rows.shape[0]:
        raise ValueError(
            f"row count mismatch: aa={aa_rows.shape[0]} di={di_rows.shape[0]}"
        )
    if aa_rows.ndim != 2 or di_rows.ndim != 2:
        raise ValueError("aa_rows and di_rows must be 2-D")
    return np.concatenate([aa_rows, di_rows], axis=1)


def build_aa_3di_concat_store(request: ConcatStoreRequest) -> EmbeddingStore:
    """Build and optionally persist a 2048-d AA∥3Di embedding store.

    Requires an AA ``EmbeddingStore`` and either a precomputed 3Di matrix
    (``di_embeddings`` + ``di_ids``) and/or 3Di sequences / PDBs for encoding.
    """
    output_dir = Path(request.output_dir)
    if (
        not request.overwrite
        and output_dir.exists()
        and (output_dir / "embeddings.npy").exists()
    ):
        raise FileExistsError(
            f"{output_dir} already populated; pass overwrite=True to rebuild"
        )

    aa = EmbeddingStore.from_dir(request.aa_store, require_labels=False)
    needed = list(dict.fromkeys(request.domain_ids))
    missing_aa = [d for d in needed if d not in aa.id_to_idx]
    if missing_aa:
        raise RuntimeError(f"{len(missing_aa)} domains missing from AA store")

    di_index: dict[str, int] = {}
    di_mat: np.ndarray | None = None
    if request.di_embeddings is not None:
        if request.di_ids is None:
            raise ValueError("di_ids required when di_embeddings is set")
        if len(request.di_ids) != request.di_embeddings.shape[0]:
            raise RuntimeError("3Di cache ids/rows mismatch")
        di_mat = np.asarray(request.di_embeddings)
        di_index = {did: i for i, did in enumerate(request.di_ids)}

    missing_di = [d for d in needed if d not in di_index]
    di_seqs = dict(request.di_sequences or {})
    if missing_di:
        still = [d for d in missing_di if d not in di_seqs]
        if still:
            if request.pdb_dir is None:
                raise RuntimeError(
                    f"{len(still)} domains lack 3Di embeddings/sequences and "
                    "pdb_dir was not provided"
                )
            foldseek = resolve_foldseek(request.foldseek)
            work = request.foldseek_work_dir or (output_dir / "_foldseek_work")
            if work.exists():
                shutil.rmtree(work)
            extracted = extract_3di_from_pdbs(
                foldseek=foldseek,
                pdb_dir=Path(request.pdb_dir),
                domain_ids=still,
                work_dir=work,
                threads=request.foldseek_threads,
            )
            di_seqs.update(extracted)

        encode_ids = [d for d in missing_di if d in di_seqs]
        absent = [d for d in missing_di if d not in di_seqs]
        if absent and len(absent) < len(needed):
            logger.warning(
                "Dropping %d domains without 3Di: %s", len(absent), absent[:5]
            )
        if encode_ids:
            cfg = request.encode_config or EncodeConfig(half=True, dtype="float16")
            with ProstT5Encoder(cfg) as enc:
                di_vecs = enc.encode(
                    {d: di_seqs[d] for d in encode_ids}, modality="3di"
                )
            new_ids = list(di_index.keys()) + list(di_vecs.keys())
            new_rows = [np.asarray(di_vecs[d], dtype=np.float16) for d in di_vecs]
            if di_mat is None:
                di_mat = np.stack(new_rows, axis=0)
            else:
                di_mat = np.concatenate(
                    [np.asarray(di_mat, dtype=np.float16), np.stack(new_rows, axis=0)],
                    axis=0,
                )
            di_index = {did: i for i, did in enumerate(new_ids)}

    kept = [d for d in needed if d in di_index]
    if needed and not kept:
        raise RuntimeError("No requested domains have usable 3Di embeddings")
    if di_mat is None:
        raise RuntimeError("no 3Di embeddings available")

    aa_rows = np.stack(
        [np.asarray(aa.embeddings[aa.id_to_idx[d]], dtype=np.float16) for d in kept],
        axis=0,
    )
    di_rows = np.stack(
        [np.asarray(di_mat[di_index[d]], dtype=np.float16) for d in kept],
        axis=0,
    )
    concat = concat_aa_di_rows(aa_rows, di_rows)
    labels = (
        np.asarray(
            [int(aa.labels[aa.id_to_idx[d]]) for d in kept],
            dtype=np.int64,
        )
        if aa.labels is not None
        else None
    )

    store = EmbeddingStore(
        embeddings=concat,
        ids=kept,
        labels=labels,
        id_to_idx={d: i for i, d in enumerate(kept)},
        idx_to_label=aa.idx_to_label,
    )
    if output_dir.exists() and request.overwrite:
        for name in CANONICAL_EMBEDDING_FILES:
            (output_dir / name).unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    store.save(
        output_dir,
        source=request.source,
        extra_metadata={
            "modality": "aa_3di_concat",
            "aa_dim": int(aa_rows.shape[1]),
            "di_dim": int(di_rows.shape[1]),
            "n_requested": len(needed),
            "n_kept": len(kept),
        },
    )
    logger.info("Wrote %s shape=%s", output_dir, concat.shape)
    return store


def _load_id_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    """CLI for ``contrasted-build-concat-store``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser(
        description="Build a 2048-d AA∥3Di ProstT5 embedding store.",
    )
    p.add_argument("--aa-store", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--domain-ids",
        type=Path,
        help="Text file with one domain id per line",
    )
    p.add_argument(
        "--split-fastas",
        type=Path,
        nargs="+",
        help="FASTA files whose domain ids define the store roster",
    )
    p.add_argument("--di-cache-npy", type=Path, default=None)
    p.add_argument("--di-cache-ids", type=Path, default=None)
    p.add_argument("--di-fasta", type=Path, default=None)
    p.add_argument("--pdb-dir", type=Path, default=None)
    p.add_argument("--foldseek", type=Path, default=None)
    p.add_argument("--foldseek-threads", type=int, default=16)
    p.add_argument("--model-name", default="Rostlab/ProstT5")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    if args.domain_ids is None and not args.split_fastas:
        p.error("provide --domain-ids and/or --split-fastas")

    domain_ids: list[str] = []
    seen: set[str] = set()
    if args.domain_ids is not None:
        for did in _load_id_list(args.domain_ids):
            if did not in seen:
                seen.add(did)
                domain_ids.append(did)
    if args.split_fastas:
        for fasta in args.split_fastas:
            for did in read_fasta_sequences(fasta):
                if did not in seen:
                    seen.add(did)
                    domain_ids.append(did)

    di_embeddings = None
    di_ids = None
    if args.di_cache_npy is not None:
        if args.di_cache_ids is None:
            p.error("--di-cache-ids required with --di-cache-npy")
        di_embeddings = np.load(args.di_cache_npy, mmap_mode="r")
        di_ids = _load_id_list(args.di_cache_ids)

    di_sequences = None
    if args.di_fasta is not None:
        di_sequences = read_fasta_sequences(args.di_fasta)

    build_aa_3di_concat_store(
        ConcatStoreRequest(
            aa_store=args.aa_store,
            output_dir=args.output_dir,
            domain_ids=domain_ids,
            di_embeddings=di_embeddings,
            di_ids=di_ids,
            di_sequences=di_sequences,
            pdb_dir=args.pdb_dir,
            foldseek=args.foldseek,
            foldseek_threads=args.foldseek_threads,
            encode_config=EncodeConfig(model_name=args.model_name, half=True),
            overwrite=bool(args.overwrite),
        )
    )


if __name__ == "__main__":
    main()
