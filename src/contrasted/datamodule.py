"""Lightning datamodule and CATH hierarchy for training.

Annotate / make_db / embed use ``contrasted.data.EmbeddingStore``. This module
is training-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from contrasted.data import (
    EmbeddingStore,
    load_domain_ids_from_fasta,
    resolve_fasta_paths,
)

logger = logging.getLogger(__name__)

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

