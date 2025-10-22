import torch
import lightning as L
import faiss
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import logging
import warnings

logger = logging.getLogger(__name__)


class KNNEvaluationCallback(L.Callback):
    """1-NN evaluation using FAISS for efficient cosine similarity search."""
    
    def __init__(
        self, 
        eval_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        
        # Cache for training embeddings and FAISS index
        self._faiss_index: Optional[faiss.Index] = None
        self._train_labels: Optional[np.ndarray] = None
        self._embedding_dim: Optional[int] = None
    
    def _build_faiss_index(
        self, 
        train_embs: torch.Tensor, 
        train_labs: torch.Tensor
    ) -> None:
        """Build FAISS index for k-NN search using cosine similarity.
        
        Args:
            train_embs: L2-normalized training embeddings [N, D]
            train_labs: Training labels [N]
        """
        # Convert to numpy and ensure float32 (FAISS requirement)
        train_embs_np = train_embs.cpu().numpy().astype('float32')
        self._train_labels = train_labs.cpu().numpy()
        self._embedding_dim = train_embs_np.shape[1]
        
        # Use IndexFlatIP (Inner Product) since embeddings are L2-normalized
        # For normalized vectors: cosine_similarity(a, b) = dot(a, b)
        self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)
        self._faiss_index.add(train_embs_np)
        
        logger.info(f"  Built FAISS index with {train_embs_np.shape[0]} train embeddings (dim={self._embedding_dim})")
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Compute k-NN metrics at validation epoch end."""
        
        if trainer.sanity_checking:
            return
        
        if trainer.current_epoch % self.eval_every_n_epochs != 0:
            return
        
        # Build FAISS index if not already built (only once)
        if self._faiss_index is None:
            logger.info(f"Building FAISS index for k-NN evaluation (one-time)")
            train_embs, train_labs = self._collect_embeddings(
                trainer, pl_module, trainer.datamodule.train_dataset
            )
            self._build_faiss_index(train_embs, train_labs)
        
        # Collect validation embeddings
        val_embeddings, val_labels = self._collect_embeddings(
            trainer, pl_module, trainer.datamodule.val_dataset
        )
        
        # Compute k-NN metrics using FAISS
        metrics = self._compute_knn_metrics(val_embeddings, val_labels)
        
        # Log metrics
        for name, value in metrics.items():
            pl_module.log(
                f"val/knn_{name}",
                value, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True,
                sync_dist=True
            )
    
    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Compute 1-NN metrics at test epoch end using train as reference."""
        
        # Ensure FAISS index is built
        if self._faiss_index is None:
            logger.info("Building FAISS index for k-NN evaluation (test phase)")
            train_embs, train_labs = self._collect_embeddings(
                trainer, pl_module, trainer.datamodule.train_dataset
            )
            self._build_faiss_index(train_embs, train_labs)
        
        # Collect test embeddings
        test_embeddings, test_labels = self._collect_embeddings(
            trainer, pl_module, trainer.datamodule.test_dataset
        )
        
        # Compute k-NN metrics using FAISS
        metrics = self._compute_knn_metrics(test_embeddings, test_labels)
        
        # Log metrics
        for name, value in metrics.items():
            pl_module.log(
                f"test/knn_{name}",
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
        dataset
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect L2-normalized embeddings from dataset.
        
        Creates a temporary single-worker dataloader to avoid file descriptor leaks
        from persistent workers with HDF5 files.
        """
        from torch.utils.data import DataLoader
        
        pl_module.eval()
        
        # Create a temporary dataloader with num_workers=0 to avoid file handle issues
        temp_dataloader = DataLoader(
            dataset,
            batch_size=trainer.datamodule.hparams.batch_size,
            shuffle=False,
            num_workers=0,  # Critical: avoid file descriptor leaks
            pin_memory=False,
        )
        
        all_embeddings = []
        all_labels = []
        
        for batch in temp_dataloader:
            embeddings, labels = batch
            embeddings = embeddings.to(pl_module.device)
            
            projected = pl_module(embeddings)
            
            all_embeddings.append(projected.cpu())
            all_labels.append(labels)
        
        embeddings_norm = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return embeddings_norm, labels
    
    def _compute_knn_metrics(
        self,
        query_embeddings: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute 1-NN metrics using FAISS index.
        
        Args:
            query_embeddings: L2-normalized query embeddings [N, D]
            query_labels: Query labels [N]
            
        Returns:
            Dictionary with accuracy, balanced_accuracy, and macro_f1
        """
        # Convert query embeddings to numpy float32
        query_embs_np = query_embeddings.cpu().numpy().astype('float32')
        query_labs_np = query_labels.cpu().numpy()
        
        # Search for k=1 nearest neighbor using FAISS
        # distances: [N, 1], indices: [N, 1]
        distances, indices = self._faiss_index.search(query_embs_np, k=1)
        
        # Get predicted labels from nearest neighbors
        nearest_indices = indices[:, 0]  # Shape: [N]
        predicted_labels = self._train_labels[nearest_indices]
        
        # Get all unique classes from both y_true and y_pred
        all_classes = sorted(set(query_labs_np) | set(predicted_labels))
        
        # Compute metrics
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
            metrics = {
                "accuracy": float(accuracy_score(query_labs_np, predicted_labels)),
                "balanced_accuracy": float(balanced_accuracy_score(query_labs_np, predicted_labels)),
                "macro_f1": float(f1_score(query_labs_np, predicted_labels, average='macro', zero_division=0, labels=all_classes)),
            }
        
        return metrics
