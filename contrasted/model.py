import torch
import torch.nn as nn
import lightning as L
from typing import List, Optional, Dict, Any

from .losses import SupConLoss, ProxyAnchorLoss


class ProjectionHead(nn.Module):
    """MLP projection head with L2-normalized outputs."""
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projector(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)


class CathSupConModel(L.LightningModule):
    """Contrastive learning model for CATH superfamily classification."""
    
    def __init__(
        self,
        input_dim: int = 1024,
        proj_hidden_dims: List[int] = [512, 256],
        proj_output_dim: int = 128,
        dropout: float = 0.1,
        loss_type: str = "supcon",
        loss_params: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        scheduler_params: Optional[Dict[str, Any]] = None,
        num_classes: Optional[int] = None,
        epochs: int = 400,
        num_warmup_epochs: int = 20,
        min_lr: float = 1e-6,
        eps: float = 1e-8,
        betas: tuple = (0.9, 0.999),
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.projection_head = ProjectionHead(
            input_dim=input_dim,
            hidden_dims=proj_hidden_dims,
            output_dim=proj_output_dim,
            dropout=dropout,
        )
        
        # Lazy initialization for proxy-anchor loss
        self._num_classes = num_classes
        
        loss_params = loss_params or {}
        loss_type = loss_type.lower()
        
        if loss_type == "supcon":
            self.main_loss = SupConLoss(
                temperature=loss_params.get("temperature", 0.07)
            )
        elif loss_type == "proxy_anchor":
            # Proxy anchor requires num_classes; will be initialized in setup()
            self._proxy_anchor_params = {
                "embedding_dim": proj_output_dim,
                "margin": loss_params.get("margin", 0.1),
                "alpha": loss_params.get("alpha", 32.0),
            }
            self.main_loss = None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.projection_head(embeddings)
    
    def setup(self, stage: str):
        """Initialize components that require num_classes from datamodule."""
        if self._num_classes is None:
            self._num_classes = self.trainer.datamodule.num_classes
        
        # Initialize proxy anchor loss if needed
        if self.main_loss is None:
            self.main_loss = ProxyAnchorLoss(
                num_classes=self._num_classes,
                **self._proxy_anchor_params
            )
    
    def _shared_step(self, batch, batch_idx, stage: str):
        embeddings, labels = batch
        projected = self(embeddings)
        
        loss = self.main_loss(projected, labels)
        self.log(
            f'{stage}/loss',
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.eps,
            betas=self.hparams.betas,
        )
        # learning rate warmup
        linear_lr = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.5, total_iters=self.hparams.num_warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.epochs - self.hparams.num_warmup_epochs,
            eta_min=self.hparams.min_lr,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[linear_lr, cosine],
            milestones=[self.hparams.num_warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
