"""Lightning callbacks for training evaluation."""

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import accuracy, f1_score, recall

from .search import VectorIndex


class KNNEvaluationCallback(L.Callback):
    """1-NN evaluation using cosine similarity search.

    Rebuilds the index on each validation epoch to reflect updated model weights.
    """

    def __init__(self, eval_every_n_epochs: int = 1):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if (
            trainer.sanity_checking
            or trainer.current_epoch % self.eval_every_n_epochs != 0
        ):
            return

        dm = trainer.datamodule  # type: ignore[union-attr]
        train_embs, train_labs = self._collect_embeddings(
            trainer, pl_module, dm.train_dataset
        )
        index = VectorIndex(train_embs, ids=None, dtype=torch.float32)

        val_embs, val_labs = self._collect_embeddings(
            trainer, pl_module, dm.val_dataset
        )
        metrics = self._compute_metrics(index, train_labs, val_embs, val_labs)

        for name, value in metrics.items():
            pl_module.log(
                f"val/knn_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        dm = trainer.datamodule  # type: ignore[union-attr]
        train_embs, train_labs = self._collect_embeddings(
            trainer, pl_module, dm.train_dataset
        )
        index = VectorIndex(train_embs, ids=None, dtype=torch.float32)

        if hasattr(dm, "test_datasets") and dm.test_datasets:
            test_sets = dm.test_datasets.items()
        else:
            test_sets = [("", dm.test_dataset)]

        for test_name, test_dataset in test_sets:
            test_embs, test_labs = self._collect_embeddings(
                trainer, pl_module, test_dataset
            )
            metrics = self._compute_metrics(index, train_labs, test_embs, test_labs)

            prefix = f"test/{test_name}/" if test_name else "test/"
            for name, value in metrics.items():
                pl_module.log(
                    f"{prefix}knn_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )

    @torch.no_grad()
    def _collect_embeddings(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        dataset,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pl_module.eval()
        dm = trainer.datamodule  # type: ignore[union-attr]
        dataloader = DataLoader(
            dataset,
            batch_size=dm.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        all_embeddings = []
        all_labels = []

        for embeddings, labels in dataloader:
            projected = pl_module(embeddings.to(pl_module.device))
            all_embeddings.append(projected.cpu())
            all_labels.append(labels)

        return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)

    def _compute_metrics(
        self,
        index: VectorIndex,
        train_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> dict[str, float]:
        indices = index.search(query_embeddings, k=1)[1].squeeze(1).cpu()
        preds = train_labels[indices]
        target = query_labels.cpu()
        num_classes = int(max(preds.max(), target.max())) + 1

        metrics = {
            "accuracy": float(
                accuracy(preds, target, task="multiclass", num_classes=num_classes)
            ),
            "balanced_accuracy": float(
                recall(
                    preds,
                    target,
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                )
            ),
            "macro_f1": float(
                f1_score(
                    preds,
                    target,
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                )
            ),
        }

        return metrics
