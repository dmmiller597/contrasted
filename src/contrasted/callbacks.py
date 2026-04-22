"""Lightning callbacks for training-time evaluation."""

import lightning as L
import torch
from torch.utils.data import DataLoader

from contrasted.search import VectorIndex
from contrasted.utils import accuracy


class KNNEvaluationCallback(L.Callback):
    """1-NN evaluation using cosine similarity search.

    Rebuilds the train-embedding index at the end of each validation epoch
    (cadence controlled by ``eval_every_n_epochs``) and logs 1-NN accuracy.
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
        dm = trainer.datamodule
        if dm is None:
            return
        acc = self._evaluate(pl_module, dm.train_dataset, dm.val_dataset, dm.batch_size)
        pl_module.log(
            "val/knn_accuracy", acc, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        dm = trainer.datamodule
        if dm is None:
            return
        if getattr(dm, "test_datasets", None):
            test_sets = list(dm.test_datasets.items())
        else:
            test_sets = [("", dm.test_dataset)]

        for test_name, test_dataset in test_sets:
            acc = self._evaluate(
                pl_module, dm.train_dataset, test_dataset, dm.batch_size
            )
            key = f"test/{test_name}/knn_accuracy" if test_name else "test/knn_accuracy"
            pl_module.log(key, acc, on_epoch=True, sync_dist=True)

    def _evaluate(
        self,
        pl_module: L.LightningModule,
        train_dataset,
        query_dataset,
        batch_size: int,
    ) -> float:
        train_embs, train_labs = self._collect_embeddings(
            pl_module, train_dataset, batch_size
        )
        index = VectorIndex(train_embs, ids=None)

        query_embs, query_labs = self._collect_embeddings(
            pl_module, query_dataset, batch_size
        )
        neighbor_idx = index.search(query_embs, k=1)[1].squeeze(1).cpu()
        preds = train_labs[neighbor_idx]
        return accuracy(preds, query_labs)

    @torch.inference_mode()
    def _collect_embeddings(
        self,
        pl_module: L.LightningModule,
        dataset,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = pl_module.training
        pl_module.eval()
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            embs: list[torch.Tensor] = []
            labs: list[torch.Tensor] = []
            for embeddings, labels in dataloader:
                projected = pl_module(embeddings.to(pl_module.device))
                embs.append(projected.cpu())
                labs.append(labels)
            return torch.cat(embs, dim=0), torch.cat(labs, dim=0)
        finally:
            pl_module.train(was_training)
