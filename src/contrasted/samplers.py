"""Batch samplers for contrastive training."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


class BalancedMPerClassBatchSampler(Sampler[list[int]]):
    """Build fixed-size batches with up to ``m_per_class`` examples per H class.

    Within each batch, classes are selected uniformly **without replacement**.
    Classes with at least two domains contribute up to ``m_per_class`` distinct
    rows; singletons contribute exactly one. The final one-sample fill, if
    needed, also uses a previously unselected class so within-batch row
    multiplicity stays ≤ ``m_per_class``.
    """

    def __init__(
        self,
        labels: Sequence[int] | np.ndarray,
        *,
        batch_size: int = 1024,
        m_per_class: int = 2,
        batches_per_epoch: int = 112,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if m_per_class <= 0:
            raise ValueError("m_per_class must be positive.")
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive.")

        labels_arr = np.asarray(labels, dtype=np.int64)
        if labels_arr.ndim != 1 or labels_arr.size == 0:
            raise ValueError("labels must be a non-empty 1-D array.")

        self.batch_size = int(batch_size)
        self.m_per_class = int(m_per_class)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self._epoch = 0
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.labels = labels_arr
        self.last_batches: list[list[int]] = []

        class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels_arr.tolist()):
            class_to_indices[int(label)].append(idx)
        self._class_ids = sorted(class_to_indices)
        self._class_to_indices = {
            cid: np.asarray(indices, dtype=np.int64)
            for cid, indices in class_to_indices.items()
        }
        if not self._class_ids:
            raise ValueError("No classes found in labels.")

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so each epoch has a deterministic but different schedule."""
        self._epoch = int(epoch)
        self._generator.manual_seed(self.seed + self._epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        self.last_batches = []
        for _ in range(self.batches_per_epoch):
            batch = self._sample_batch()
            self.last_batches.append(batch)
            yield batch

    def _sample_batch(self) -> list[int]:
        selected: list[int] = []
        used_classes: set[int] = set()

        while len(selected) < self.batch_size:
            remaining = self.batch_size - len(selected)
            available = [c for c in self._class_ids if c not in used_classes]
            if not available:
                raise RuntimeError(
                    "Cannot fill batch without reuse: fewer eligible classes "
                    f"than needed (batch_size={self.batch_size})."
                )

            choice_idx = int(
                torch.randint(len(available), (1,), generator=self._generator).item()
            )
            class_id = available[choice_idx]
            used_classes.add(class_id)

            members = self._class_to_indices[class_id]
            if len(members) == 1:
                take = 1
            elif remaining == 1:
                take = 1
            else:
                take = min(self.m_per_class, len(members), remaining)

            if take == len(members):
                chosen = members.tolist()
            else:
                perm = torch.randperm(len(members), generator=self._generator).tolist()
                chosen = members[perm[:take]].tolist()
            selected.extend(chosen)

        return selected

    def schedule_stats(
        self, batches: list[list[int]] | None = None
    ) -> dict[str, object]:
        """Return provenance statistics for ``batches`` (default: last epoch)."""
        if batches is None:
            batches = self.last_batches
        if not batches:
            raise RuntimeError("No batches available for schedule_stats().")

        flat = [idx for batch in batches for idx in batch]
        payload = ",".join(str(i) for i in flat).encode()
        class_counts: dict[int, int] = defaultdict(int)
        singleton_anchors = 0
        valid_supcon_anchors = 0
        max_multiplicity = 0

        for batch in batches:
            counts: dict[int, int] = defaultdict(int)
            row_counts: dict[int, int] = defaultdict(int)
            for idx in batch:
                lab = int(self.labels[idx])
                counts[lab] += 1
                class_counts[lab] += 1
                row_counts[idx] += 1
            max_multiplicity = max(max_multiplicity, max(row_counts.values()))
            for lab, n in counts.items():
                if n == 1 and len(self._class_to_indices[lab]) == 1:
                    singleton_anchors += n
                if n >= 2:
                    valid_supcon_anchors += n

        n_positions = len(flat)
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "num_batches": len(batches),
            "num_unique_classes": len(class_counts),
            "singleton_anchor_count": singleton_anchors,
            "valid_supcon_anchor_fraction": (
                valid_supcon_anchors / n_positions if n_positions else 0.0
            ),
            "max_within_batch_row_multiplicity": max_multiplicity,
        }

    def materialize_epoch(self, epoch: int) -> list[list[int]]:
        """Build all batches for ``epoch`` without permanently mutating iter state."""
        saved = self._epoch
        self.set_epoch(epoch)
        batches = [self._sample_batch() for _ in range(self.batches_per_epoch)]
        self.set_epoch(saved)
        return batches
