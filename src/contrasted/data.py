"""Data loading for protein embeddings from embedding directories."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from contrasted.utils import require_exists

logger = logging.getLogger(__name__)

CANONICAL_EMBEDDING_FILES = (
    "embeddings.npy",
    "labels.npy",
    "ids.txt",
    "metadata.json",
)

# Expected invariants for the CATH S40 objective-matrix split.
S40_SPLIT_EXPECTATIONS = {
    "train_domains": 114_350,
    "train_h": 5_324,
    "val_queries": 396,
    "test_queries": 769,
    "num_c": 3,
    "num_a": 40,
    "num_t": 1_225,
    "num_h": 5_324,
}


def extract_cath_level(cath_code: str, level: str) -> str:
    """Extract a CATH hierarchy prefix from a ``C.A.T.H`` code."""
    parts = str(cath_code).split(".")
    if level == "C":
        return parts[0]
    if level == "A":
        return ".".join(parts[:2])
    if level == "T":
        return ".".join(parts[:3])
    if level == "H":
        return str(cath_code)
    raise ValueError(f"Unknown CATH level: {level!r}")


@dataclass(frozen=True)
class CathHierarchy:
    """Dense H-index lookups into C/A/T/H codes and ancestor indices."""

    levels: tuple[str, ...]
    num_classes: dict[str, int]
    h_to_level: dict[str, torch.Tensor]
    idx_to_code: dict[str, tuple[str, ...]]


def build_cath_hierarchy(h_codes: list[str]) -> CathHierarchy:
    """Build hierarchy banks from dense-ordered H-level CATH codes."""
    if not h_codes:
        raise ValueError("h_codes must be non-empty.")

    level_to_codes: dict[str, list[str]] = {"C": [], "A": [], "T": [], "H": []}
    level_to_index: dict[str, dict[str, int]] = {
        "C": {},
        "A": {},
        "T": {},
        "H": {},
    }
    for h_code in h_codes:
        for level in ("C", "A", "T", "H"):
            code = extract_cath_level(h_code, level)
            if code not in level_to_index[level]:
                level_to_index[level][code] = len(level_to_codes[level])
                level_to_codes[level].append(code)

    n_h = len(h_codes)
    h_to_level = {
        "C": torch.empty(n_h, dtype=torch.long),
        "A": torch.empty(n_h, dtype=torch.long),
        "T": torch.empty(n_h, dtype=torch.long),
        "H": torch.arange(n_h, dtype=torch.long),
    }
    for h_idx, h_code in enumerate(h_codes):
        h_to_level["C"][h_idx] = level_to_index["C"][extract_cath_level(h_code, "C")]
        h_to_level["A"][h_idx] = level_to_index["A"][extract_cath_level(h_code, "A")]
        h_to_level["T"][h_idx] = level_to_index["T"][extract_cath_level(h_code, "T")]

    return CathHierarchy(
        levels=("C", "A", "T", "H"),
        num_classes={level: len(codes) for level, codes in level_to_codes.items()},
        h_to_level=h_to_level,
        idx_to_code={level: tuple(codes) for level, codes in level_to_codes.items()},
    )


def parse_fasta_header(header: str) -> str:
    """Extract a domain/sequence ID from a FASTA header line.

    Recognized formats:
    - CATH: ``>cath|{cath_release}|{domain_id}/{start}-{end}`` -> ``domain_id``
    - Fallback (including TED/AlphaFold ``>AF-..._TED03``): first whitespace
      token after ``>``, with any trailing ``/start-end`` stripped.
    """
    header = header.strip()
    if not header.startswith(">"):
        raise ValueError(f"Invalid FASTA header: {header}")

    tokens = header[1:].split()
    if not tokens:
        # A bare ">" (no id). Raise ValueError so callers skip/warn consistently
        # rather than crashing on an uncaught IndexError.
        raise ValueError(f"Empty FASTA header: {header!r}")
    token = tokens[0]
    parts = token.split("|")
    if parts[0] == "cath" and len(parts) >= 3:
        return parts[2].split("/", 1)[0]
    return token.split("/", 1)[0]


def load_domain_ids_from_fasta(fasta_path: str | Path) -> list[str]:
    """Read a FASTA file and return the list of parsed domain IDs (in order)."""
    domain_ids: list[str] = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                try:
                    domain_ids.append(parse_fasta_header(line))
                except ValueError as e:
                    logger.warning(f"Could not parse header: {line.strip()} - {e}")
    return domain_ids


def read_fasta_sequences(fasta_path: str | Path) -> dict[str, str]:
    """Read a FASTA file into an ordered ``{domain_id: sequence}`` map.

    Gaps (``-``) are stripped and multi-line sequences are joined. Duplicate
    IDs keep the first occurrence and emit a warning.
    """
    # Accumulate residue lines per id and join once, rather than repeated
    # string concatenation (quadratic for long, many-line records).
    chunks: dict[str, list[str]] = {}
    current_id: str | None = None
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                try:
                    current_id = parse_fasta_header(line)
                except ValueError as e:
                    logger.warning(f"Skipping invalid header: {e}")
                    current_id = None
                    continue
                if current_id in chunks:
                    logger.warning(f"Duplicate ID '{current_id}'; keeping first")
                    current_id = None
                else:
                    chunks[current_id] = []
            elif current_id is not None:
                chunks[current_id].append(line.replace("-", ""))
    sequences = {did: "".join(parts) for did, parts in chunks.items()}
    for did, seq in sequences.items():
        if not seq:
            raise ValueError(
                f"FASTA record '{did}' has no residues "
                f"(header-only or gap-only sequence in {fasta_path})"
            )
    return sequences


def resolve_fasta_paths(fasta_input: str | Path) -> dict[str, Path]:
    """Resolve ``fasta_input`` to a dict mapping split name -> path.

    Accepts a single file or a directory; returns ``{}`` if nothing is found.
    """
    fasta_input = Path(fasta_input)
    if fasta_input.is_dir():
        fasta_files = sorted(fasta_input.glob("*.fasta")) + sorted(
            fasta_input.glob("*.fa")
        )
        if not fasta_files:
            logger.warning(f"No FASTA files found in directory: {fasta_input}")
            return {}
        logger.info(f"Found {len(fasta_files)} FASTA files in {fasta_input}")
        return {f.stem: f for f in fasta_files}
    if fasta_input.is_file():
        return {fasta_input.stem: fasta_input}
    logger.warning(f"FASTA path not found: {fasta_input}")
    return {}


# ---------------------------------------------------------------------------
# Embedding directory loading
# ---------------------------------------------------------------------------


def read_ids_txt(path: Path) -> list[str]:
    """Domain ids from a store's ``ids.txt``, in row order.

    Line *i* is authoritative for ``embeddings.npy`` row *i*, so this is part of
    the store format contract rather than an internal helper.
    """
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# Retained because callers inside this module use the private spelling.
_read_ids_txt = read_ids_txt


def _load_idx_to_label(embedding_dir: Path, metadata: dict) -> dict[int, str] | None:
    if idx_to_label := metadata.get("idx_to_label"):
        if isinstance(idx_to_label, dict):
            return {int(k): v for k, v in idx_to_label.items()}

    if (sidecar := metadata.get("idx_to_label_file")) and (
        path := embedding_dir / sidecar
    ).exists():
        with open(path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return None


# ---------------------------------------------------------------------------
# EmbeddingStore: the single entry point for embedding-directory I/O
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingStore:
    """Owns the contents of an embedding directory.

    The directory must contain:
    - ``embeddings.npy``: ``(N, D)`` array (memory-mapped on load)
    - ``ids.txt``: one ID per line
    - ``metadata.json``: at minimum ``dims``/``count``/``dtype``
    - ``labels.npy`` (optional): ``(N,)`` int64 class indices

    The ``ids.txt`` line order is authoritative: row ``i`` of ``embeddings.npy``
    is domain ``ids[i]``, so the ID -> row mapping is rebuilt from ``ids.txt``.
    """

    embeddings: np.ndarray
    ids: list[str]
    labels: np.ndarray | None
    id_to_idx: dict[str, int]
    idx_to_label: dict[int, str] | None = None
    embedding_dir: Path | None = field(default=None, compare=False)
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        *,
        require_labels: bool = False,
    ) -> "EmbeddingStore":
        path = Path(path)
        embeddings_path = path / "embeddings.npy"
        ids_path = path / "ids.txt"
        metadata_path = path / "metadata.json"
        labels_path = path / "labels.npy"

        for required, name in [
            (embeddings_path, "embeddings.npy"),
            (ids_path, "ids.txt"),
            (metadata_path, "metadata.json"),
        ]:
            require_exists(required, name)

        with open(metadata_path) as f:
            metadata = json.load(f)

        embeddings = np.load(embeddings_path, mmap_mode="r")

        for key, actual, cast in [
            ("dtype", embeddings.dtype, np.dtype),
            ("dims", embeddings.shape[1], int),
            ("count", embeddings.shape[0], int),
        ]:
            if (expected := metadata.get(key)) is not None and actual != cast(expected):
                raise ValueError(
                    f"embeddings.npy {key} {actual} != metadata {key} {expected}"
                )

        ids = _read_ids_txt(ids_path)
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"ids.txt has {len(ids)} entries but embeddings.npy has "
                f"{embeddings.shape[0]}"
            )

        labels: np.ndarray | None = None
        if labels_path.exists():
            labels = np.asarray(np.load(labels_path), dtype=np.int64)
            if labels.shape[0] != embeddings.shape[0]:
                raise ValueError(
                    f"labels.npy has {labels.shape[0]} entries but "
                    f"embeddings.npy has {embeddings.shape[0]}"
                )
        elif require_labels:
            raise FileNotFoundError(f"Missing labels.npy in {path}")

        idx_to_label = _load_idx_to_label(path, metadata)

        return cls(
            embeddings=embeddings,
            ids=ids,
            labels=labels,
            id_to_idx={id_: i for i, id_ in enumerate(ids)},
            idx_to_label=idx_to_label,
            embedding_dir=path,
            metadata=metadata,
        )

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.idx_to_label) if self.idx_to_label else 0

    def resolve(self, domain_ids: list[str]) -> tuple[list[int], list[str], list[str]]:
        """Map domain IDs to embedding-row indices.

        Returns ``(found_indices, found_ids, missing_ids)`` in the same
        order as ``domain_ids``.
        """
        found_idx, found_ids, missing_ids = [], [], []
        for d in domain_ids:
            if (row := self.id_to_idx.get(d)) is not None:
                found_idx.append(row)
                found_ids.append(d)
            else:
                missing_ids.append(d)
        return found_idx, found_ids, missing_ids

    def get_batch(
        self, domain_ids: list[str]
    ) -> tuple[torch.Tensor, list[str], list[str]]:
        """Fetch embeddings for ``domain_ids`` as a ``(M, D)`` float tensor."""
        found_idx, found_ids, missing_ids = self.resolve(domain_ids)
        batch = (
            torch.from_numpy(np.array(self.embeddings[found_idx], copy=True)).float()
            if found_idx
            else torch.empty(0, self.dim)
        )
        return batch, found_ids, missing_ids

    def save(
        self,
        path: str | Path,
        *,
        source: str | None = None,
        extra_metadata: dict | None = None,
    ) -> Path:
        """Write the canonical 4-file layout to ``path``.

        Writes are atomic per-file: each file is staged as ``<name>.tmp`` in
        the target directory and renamed on success. ``path`` must not
        already contain any of the canonical files.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        existing = [n for n in CANONICAL_EMBEDDING_FILES if (path / n).exists()]
        if existing:
            raise FileExistsError(
                f"{path} already contains {existing}; refusing to overwrite"
            )

        def _atomic(name: str, writer):
            tmp = path / f"{name}.tmp"
            writer(tmp)
            tmp.replace(path / name)

        def _save_npy(path: Path, value) -> None:
            with path.open("wb") as f:
                np.save(f, value)

        embeddings = np.ascontiguousarray(self.embeddings)
        _atomic("embeddings.npy", lambda p: _save_npy(p, embeddings))

        if self.labels is not None:
            _atomic("labels.npy", lambda p: _save_npy(p, np.asarray(self.labels)))

        _atomic(
            "ids.txt",
            lambda p: p.write_text("\n".join(self.ids) + "\n"),
        )

        metadata = {
            "dims": int(embeddings.shape[1]),
            "count": int(embeddings.shape[0]),
            "dtype": str(embeddings.dtype),
            **({"source": source} if source is not None else {}),
            **(
                {"idx_to_label": {int(k): v for k, v in self.idx_to_label.items()}}
                if self.idx_to_label
                else {}
            ),
            **(extra_metadata or {}),
        }
        _atomic("metadata.json", lambda p: p.write_text(json.dumps(metadata)))

        self.embedding_dir = path
        self.metadata = metadata
        return path


# ---------------------------------------------------------------------------
# Store resolution and projection-head compatibility
# ---------------------------------------------------------------------------


def resolve_store(
    *,
    embedding_dir: str | Path | None,
    require_labels: bool = False,
) -> "EmbeddingStore":
    """Load a populated embedding store from ``embedding_dir``.

    Missing, empty, and partially populated directories raise rather than
    falling back to in-memory AA encoding.
    """
    if embedding_dir is None or not str(embedding_dir).strip():
        raise ValueError(
            "embedding_dir is required. Pass a store built by "
            "contrasted-build-concat-store (2048-d AA∥3Di) or "
            "contrasted-embed (1024-d AA)."
        )

    path = Path(embedding_dir)
    if _is_populated_store(path):
        return EmbeddingStore.from_dir(path, require_labels=require_labels)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"{path} exists but is not a valid embedding directory. "
            "Remove it or point embedding_dir elsewhere."
        )
    raise FileNotFoundError(
        f"Embedding store not found at {path}. Build one with "
        "contrasted-build-concat-store (2048-d AA∥3Di) or contrasted-embed "
        "(1024-d AA)."
    )


def validate_store_for_projection_head(
    store: EmbeddingStore,
    *,
    input_dim: int,
) -> None:
    """Reject explicit store metadata incompatible with a projection head."""
    modality = store.metadata.get("modality")
    if input_dim == 2048 and modality is not None and modality != "aa_3di_concat":
        raise ValueError(
            f"Projection head input_dim 2048 requires an aa_3di_concat store, "
            f"but metadata modality is {modality!r}. Build the store with "
            "contrasted-build-concat-store."
        )


def _is_populated_store(path: Path) -> bool:
    required = ("embeddings.npy", "ids.txt", "metadata.json")
    return path.is_dir() and all((path / name).exists() for name in required)


# ---------------------------------------------------------------------------
# Dataset / DataModule
# ---------------------------------------------------------------------------


class EmbeddingDataset(Dataset[tuple[torch.Tensor, int]]):
    """A subset of an embedding array, defined by row indices."""

    def __init__(
        self,
        embeddings: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        indices: list[int],
    ):
        self.embeddings = embeddings
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        data_idx = self.indices[index]
        embedding = self.embeddings[data_idx]
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(np.array(embedding, copy=True)).float()
        return embedding, int(self.labels[data_idx])


class EmbeddingDataModule(L.LightningDataModule):
    """Lightning data module backed by an ``EmbeddingStore``.

    Each split (train/val/test) is defined by a FASTA file whose headers
    name domains present in ``embedding_dir``. ``test_fasta`` may be a
    single file, a directory of FASTAs, or a list of paths; each becomes
    one entry in ``test_datasets``.

    Class indices for the loss are always the dense ``0..K-1`` set of
    superfamilies present in ``train_fasta`` — not the embedding store's
    global label vocabulary.
    """

    def __init__(
        self,
        train_fasta: str,
        val_fasta: str,
        test_fasta: str | list[str],
        embedding_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        knn_eval_batch_size: int | None = None,
        balanced_sampler: bool = False,
        m_per_class: int = 2,
        batches_per_epoch: int = 112,
        sampler_seed: int | None = None,
        strict_split_checks: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self._labels: np.ndarray | None = None
        self._num_classes = 0
        self._hierarchy: CathHierarchy | None = None
        self._idx_to_h_code: tuple[str, ...] = ()
        self._train_batch_sampler = None

        self.train_fasta = Path(train_fasta)
        self.val_fasta = Path(val_fasta)
        self.test_fasta_paths = self._resolve_test_paths(test_fasta)
        self.embedding_dir = Path(embedding_dir)
        self.pin_memory = pin_memory and not torch.backends.mps.is_available()
        self.knn_eval_batch_size = knn_eval_batch_size
        self.balanced_sampler = balanced_sampler
        self.m_per_class = m_per_class
        self.batches_per_epoch = batches_per_epoch
        self.sampler_seed = sampler_seed
        self.strict_split_checks = strict_split_checks

        self.store: EmbeddingStore | None = None
        self.test_datasets: dict[str, EmbeddingDataset] = {}

    @staticmethod
    def _resolve_test_paths(
        test_fasta: str | list[str],
    ) -> dict[str, Path]:
        if isinstance(test_fasta, list):
            return {Path(p).stem: Path(p) for p in test_fasta}
        return resolve_fasta_paths(Path(test_fasta))

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def hierarchy(self) -> CathHierarchy:
        if self._hierarchy is None:
            raise RuntimeError("Hierarchy has not been built; call setup() first.")
        return self._hierarchy

    @property
    def idx_to_h_code(self) -> tuple[str, ...]:
        return self._idx_to_h_code

    @property
    def train_batch_sampler(self):
        return self._train_batch_sampler

    def _use_train_classes(self) -> None:
        """Set class IDs to the dense set of superfamilies in ``train_fasta``."""
        assert self.store is not None and self.store.labels is not None
        store_labels = np.asarray(self.store.labels)
        train_ids = load_domain_ids_from_fasta(self.train_fasta)
        train_idx, _, _ = self.store.resolve(train_ids)
        if not len(train_idx):
            raise ValueError(
                f"{self.train_fasta.name}: no domains found in the embedding store."
            )
        train_classes = np.unique(store_labels[train_idx])
        remap = np.full(int(store_labels.max()) + 1, -1, dtype=np.int64)
        remap[train_classes] = np.arange(len(train_classes), dtype=np.int64)
        self._labels = remap[store_labels]
        self._num_classes = len(train_classes)

        if self.store.idx_to_label is None:
            raise ValueError("Embedding store is missing idx_to_label mapping.")
        h_codes = [str(self.store.idx_to_label[int(c)]) for c in train_classes]
        self._idx_to_h_code = tuple(h_codes)
        self._hierarchy = build_cath_hierarchy(h_codes)
        logger.info(
            "Using %d training classes (store has %d label ids); "
            "hierarchy C/A/T/H = %s",
            self._num_classes,
            self.store.num_classes,
            {
                level: self._hierarchy.num_classes[level]
                for level in self._hierarchy.levels
            },
        )

    def validate_split_invariants(self) -> None:
        """Fail if the loaded split does not match S40 objective-matrix expectations."""
        if not self.strict_split_checks:
            return
        if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):
            raise RuntimeError("Call setup('fit') before validate_split_invariants().")

        train_n = len(self.train_dataset)
        val_n = len(self.val_dataset)
        if self.test_datasets:
            test_n = sum(len(ds) for ds in self.test_datasets.values())
        else:
            test_n = None
        hierarchy = self.hierarchy
        exp = S40_SPLIT_EXPECTATIONS

        errors: list[str] = []
        if train_n != exp["train_domains"]:
            errors.append(f"train domains {train_n} != {exp['train_domains']}")
        if self.num_classes != exp["train_h"]:
            errors.append(f"train H {self.num_classes} != {exp['train_h']}")
        if val_n != exp["val_queries"]:
            errors.append(f"val queries {val_n} != {exp['val_queries']}")
        if test_n is not None and test_n != exp["test_queries"]:
            errors.append(f"test queries {test_n} != {exp['test_queries']}")
        for key, level in (
            ("num_c", "C"),
            ("num_a", "A"),
            ("num_t", "T"),
            ("num_h", "H"),
        ):
            actual = hierarchy.num_classes[level]
            if actual != exp[key]:
                errors.append(f"{level} bank {actual} != {exp[key]}")

        labels = self._labels
        if labels is None:
            raise RuntimeError("labels required for split invariant checks")

        train_h = {
            int(labels[i])
            for i in self.train_dataset.indices  # type: ignore[union-attr]
        }
        val_h = {int(labels[i]) for i in self.val_dataset.indices}  # type: ignore[union-attr]
        if not val_h.issubset(train_h):
            errors.append("validation H classes not subset of training")

        if self.test_datasets:
            test_h: set[int] = set()
            for ds in self.test_datasets.values():
                test_h.update(int(labels[i]) for i in ds.indices)  # type: ignore[union-attr]
            if val_h & test_h:
                errors.append("validation and test H-class overlap is non-zero")
            if not test_h.issubset(train_h):
                errors.append("test H classes not subset of training")

        if errors:
            raise ValueError(
                "S40 split invariant checks failed:\n- " + "\n- ".join(errors)
            )

    def setup(self, stage: str | None = None):
        if self.store is None:
            self._load_store()
        if self._labels is None:
            self._use_train_classes()

        if stage in ("fit", None):
            self.train_dataset = self._create_dataset(self.train_fasta)
            self.val_dataset = self._create_dataset(self.val_fasta)
            self._maybe_build_train_sampler()

        if stage in ("test", None):
            self.test_datasets = {
                name: self._create_dataset(path)
                for name, path in self.test_fasta_paths.items()
            }

        if self.strict_split_checks:
            # Load test datasets for overlap checks even during fit-only setup.
            if stage == "fit" and not self.test_datasets:
                self.test_datasets = {
                    name: self._create_dataset(path)
                    for name, path in self.test_fasta_paths.items()
                }
            self.validate_split_invariants()

    def _maybe_build_train_sampler(self) -> None:
        if not self.balanced_sampler:
            self._train_batch_sampler = None
            return
        from contrasted.samplers import BalancedMPerClassBatchSampler

        assert self._labels is not None
        labels = [int(self._labels[i]) for i in self.train_dataset.indices]
        seed = 0 if self.sampler_seed is None else int(self.sampler_seed)
        self._train_batch_sampler = BalancedMPerClassBatchSampler(
            labels,
            batch_size=self.batch_size,
            m_per_class=self.m_per_class,
            batches_per_epoch=self.batches_per_epoch,
            seed=seed,
        )

    def _load_store(self) -> None:
        logger.info(f"Loading embeddings from: {self.embedding_dir}")
        self.store = EmbeddingStore.from_dir(self.embedding_dir, require_labels=True)
        if self.store.idx_to_label is None and self.store.labels is not None:
            if self.store.labels.min() < 0:
                raise ValueError(
                    f"labels.npy in {self.embedding_dir} has negative values; "
                    "class indices must be non-negative."
                )
            self.store.idx_to_label = {
                i: str(i) for i in range(int(self.store.labels.max()) + 1)
            }
        logger.info(
            f"Loaded {len(self.store.ids)} embeddings, "
            f"{self.store.dim} dims, {self.store.num_classes} store label ids"
        )

    def _create_dataset(self, fasta_path: Path) -> EmbeddingDataset:
        if self.store is None or self._labels is None:
            raise RuntimeError(
                "Embedding store with train classes has not been set up."
            )

        domain_ids = load_domain_ids_from_fasta(fasta_path)
        indices, _, missing_ids = self.store.resolve(domain_ids)

        if missing_ids:
            logger.warning(
                f"{fasta_path.name}: {len(missing_ids)}/{len(domain_ids)} "
                "domains not found"
            )
        if not indices:
            raise ValueError(
                f"{fasta_path.name}: No valid samples found. "
                f"All {len(domain_ids)} domains are missing from "
                "embedding directory."
            )
        logger.info(f"{fasta_path.name}: {len(indices)} samples")

        unmapped = sum(1 for i in indices if self._labels[i] < 0)
        if unmapped:
            raise ValueError(
                f"{fasta_path.name}: {unmapped} domains have a superfamily "
                "absent from the training split."
            )
        return EmbeddingDataset(self.store.embeddings, self._labels, indices)

    def train_dataloader(self) -> DataLoader:
        if self._train_batch_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_sampler=self._train_batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.num_workers > 0,
            )
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        loaders = [self._dataloader(ds) for ds in self.test_datasets.values()]
        return loaders[0] if len(loaders) == 1 else loaders

    def _dataloader(self, dataset: Dataset, *, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
