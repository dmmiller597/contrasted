import torch
import torch.nn.functional as F
import lightning as L
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import logging
import warnings

logger = logging.getLogger(__name__)


class KNNEvaluationCallback(L.Callback):
    """1-NN evaluation using cosine similarity. Caches train embeddings (GPU or CPU)."""
    
    def __init__(
        self, 
        eval_every_n_epochs: int = 1,
        chunk_size: int = 2048,
    ):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.chunk_size = chunk_size
        
        # Cache for normalized training embeddings
        self._train_embeddings_norm: Optional[torch.Tensor] = None
        self._train_labels: Optional[torch.Tensor] = None
        self._cache_device: Optional[torch.device] = None
        self._last_cache_epoch: int = -1
    
    def _cache_train_embeddings_with_fallback(
        self, 
        train_embs: torch.Tensor, 
        train_labs: torch.Tensor, 
        device: torch.device
    ) -> None:
        """Cache training embeddings on GPU with CPU fallback on OOM.
        
        Args:
            train_embs: Training embeddings to cache
            train_labs: Training labels to cache
            device: Target device (usually GPU)
        """
        try:
            self._train_embeddings_norm = train_embs.to(device)
            self._train_labels = train_labs.to(device)
            self._cache_device = device
            logger.info(f"  Cached {train_embs.shape[0]} train embeddings on GPU")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                self._train_embeddings_norm = train_embs
                self._train_labels = train_labs
                self._cache_device = torch.device('cpu')
                logger.info(f"  Cached {train_embs.shape[0]} train embeddings on CPU (GPU OOM)")
            else:
                raise
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Compute k-NN metrics at validation epoch end."""
        
        if trainer.sanity_checking:
            return
        
        if trainer.current_epoch % self.eval_every_n_epochs != 0:
            return
        
        device = pl_module.device
        
        # Update training embeddings cache if needed
        if self._last_cache_epoch != trainer.current_epoch:
            logger.info(f"Caching training embeddings for k-NN evaluation (epoch {trainer.current_epoch})")
            train_embs, train_labs = self._collect_embeddings(
                trainer, pl_module, trainer.datamodule.train_dataloader()
            )
            self._cache_train_embeddings_with_fallback(train_embs, train_labs, device)
            self._last_cache_epoch = trainer.current_epoch
        
        # Collect validation embeddings
        val_embeddings, val_labels = self._collect_embeddings(
            trainer, pl_module, trainer.datamodule.val_dataloader()
        )
        
        # Compute k-NN metrics
        metrics = self._compute_knn_metrics(
            self._train_embeddings_norm,
            self._train_labels,
            val_embeddings,
            val_labels,
            device
        )
        
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
        device = pl_module.device
        
        # Ensure training embeddings are cached
        if self._train_embeddings_norm is None or self._train_labels is None:
            logger.info("Caching training embeddings for k-NN evaluation (test phase)")
            train_embs, train_labs = self._collect_embeddings(
                trainer, pl_module, trainer.datamodule.train_dataloader()
            )
            self._cache_train_embeddings_with_fallback(train_embs, train_labs, device)
        
        # Collect test embeddings
        test_embeddings, test_labels = self._collect_embeddings(
            trainer, pl_module, trainer.datamodule.test_dataloader()
        )
        
        # Compute k-NN metrics
        metrics = self._compute_knn_metrics(
            self._train_embeddings_norm,
            self._train_labels,
            test_embeddings,
            test_labels,
            device
        )
        
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
        dataloader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collect L2-normalized embeddings from dataloader."""
        pl_module.eval()
        
        all_embeddings = []
        all_labels = []
        
        for batch in dataloader:
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
        ref_embeddings_norm: torch.Tensor,
        ref_labels: torch.Tensor,
        query_embeddings_norm: torch.Tensor,
        query_labels: torch.Tensor,
        device: torch.device
    ) -> Dict[str, float]:
        """Compute 1-NN metrics using cosine similarity in chunks."""
        n_queries = query_embeddings_norm.shape[0]
        
        # Move reference to GPU if not already there
        if ref_embeddings_norm.device != device:
            ref_embeddings_gpu = ref_embeddings_norm.to(device)
            ref_labels_gpu = ref_labels.to(device)
        else:
            ref_embeddings_gpu = ref_embeddings_norm
            ref_labels_gpu = ref_labels
        
        # Collect predictions in chunks
        all_predictions = []
        
        for start_idx in range(0, n_queries, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, n_queries)
            query_chunk = query_embeddings_norm[start_idx:end_idx].to(device)
            
            # Cosine similarity via F.linear: query @ ref.T
            cos_sim = F.linear(query_chunk, ref_embeddings_gpu)
            
            # Get nearest neighbor (highest similarity)
            nearest_idx = cos_sim.argmax(dim=1)
            
            # Get predicted labels
            if ref_labels_gpu.device.type == 'cpu':
                chunk_predictions = ref_labels_gpu[nearest_idx.cpu()]
            else:
                chunk_predictions = ref_labels_gpu[nearest_idx].cpu()
            
            all_predictions.append(chunk_predictions)
        
        predicted_labels = torch.cat(all_predictions, dim=0)
        
        # Convert to numpy for sklearn metrics
        y_true = query_labels.numpy()
        y_pred = predicted_labels.numpy()
        
        # Get all unique classes from both y_true and y_pred
        all_classes = sorted(set(y_true) | set(y_pred))
        
        # Compute metrics
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average='macro', zero_division=0, labels=all_classes)),
            }
        
        return metrics
