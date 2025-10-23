import torch
import lightning as L
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import warnings

from contrasted.faiss_utils import build_faiss_index, search_faiss_index



class KNNEvaluationCallback(L.Callback):
    """1-NN evaluation using FAISS for efficient cosine similarity search.
    
    Rebuilds the index on each validation epoch to reflect updated model weights.
    """
    
    def __init__(self, eval_every_n_epochs: int = 1):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute k-NN metrics at validation epoch end.
        
        Rebuilds the FAISS index from scratch to reflect updated model weights.
        """
        if trainer.sanity_checking or trainer.current_epoch % self.eval_every_n_epochs != 0:
            return
        
        # Collect and build index with current model weights
        train_embs, train_labs = self._collect_embeddings(
            trainer, pl_module, trainer.datamodule.train_dataset
        )
        faiss_index = build_faiss_index(train_embs.cpu().numpy().astype(np.float32))
        
        # Evaluate on validation set
        val_embs, val_labs = self._collect_embeddings(
            trainer, pl_module, trainer.datamodule.val_dataset
        )
        metrics = self._compute_metrics(faiss_index, train_labs, val_embs, val_labs)
        
        for name, value in metrics.items():
            pl_module.log(
                f"val/knn_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
    
    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute 1-NN metrics at test epoch end.
        
        Rebuilds the FAISS index from scratch with final model weights.
        """
        # Build index from training data
        train_embs, train_labs = self._collect_embeddings(
            trainer, pl_module, trainer.datamodule.train_dataset
        )
        faiss_index = build_faiss_index(train_embs.cpu().numpy().astype(np.float32))
        
        # Determine test datasets to evaluate
        if hasattr(trainer.datamodule, 'test_datasets') and trainer.datamodule.test_datasets:
            test_sets = trainer.datamodule.test_datasets.items()
        else:
            test_sets = [("", trainer.datamodule.test_dataset)]
        
        # Evaluate each test dataset
        for test_name, test_dataset in test_sets:
            test_embs, test_labs = self._collect_embeddings(
                trainer, pl_module, test_dataset
            )
            metrics = self._compute_metrics(faiss_index, train_labs, test_embs, test_labs)
            
            # Log metrics with appropriate prefix
            prefix = f"test/{test_name}/" if test_name else "test/"
            for name, value in metrics.items():
                pl_module.log(
                    f"{prefix}knn_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )
    
    @torch.no_grad()
    def _collect_embeddings(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        dataset,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect embeddings from dataset using temporary dataloader.
        
        Returns embeddings on CPU for index building.
        """
        from torch.utils.data import DataLoader
        
        pl_module.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=trainer.datamodule.hparams.batch_size,
            shuffle=False,
            num_workers=0,  # Avoid file descriptor leaks with HDF5
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
        faiss_index: object,
        train_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute 1-NN metrics using FAISS index.
        
        Args:
            faiss_index: FAISS index built from current training embeddings
            train_labels: Training labels corresponding to index vectors
            query_embeddings: Query embeddings to evaluate
            query_labels: Ground truth labels for queries
            
        Returns:
            Dictionary with accuracy, balanced_accuracy, and macro_f1
        """
        query_np = query_embeddings.cpu().numpy().astype(np.float32)
        query_labs = query_labels.cpu().numpy()
        train_labs = train_labels.cpu().numpy()
        
        # 1-NN search
        similarities, indices = search_faiss_index(faiss_index, query_np, k=1)
        nearest_labels = train_labs[indices[:, 0]]
        
        # Get all unique classes
        all_classes = sorted(set(query_labs) | set(nearest_labels))
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            metrics = {
                "accuracy": float(accuracy_score(query_labs, nearest_labels)),
                "balanced_accuracy": float(balanced_accuracy_score(query_labs, nearest_labels)),
                "macro_f1": float(
                    f1_score(
                        query_labs,
                        nearest_labels,
                        average="macro",
                        zero_division=0,
                        labels=all_classes,
                    )
                ),
            }
        
        return metrics
