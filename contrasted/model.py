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
                layers.append(nn.ReLU(inplace=True))
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
        loss_weight: float = 1.0,
        loss_params: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        scheduler_type: str = "onecycle",
        scheduler_params: Optional[Dict[str, Any]] = None,
        warmup_epochs: int = 0,
        num_classes: Optional[int] = None,
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
            self.loss_name = "supcon"
        elif loss_type == "proxy_anchor":
            # Proxy anchor requires num_classes; will be initialized in setup()
            self._proxy_anchor_params = {
                "embedding_dim": proj_output_dim,
                "margin": loss_params.get("margin", 0.1),
                "alpha": loss_params.get("alpha", 32.0),
            }
            self.main_loss = None
            self.loss_name = "proxy_anchor"
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.loss_weight = loss_weight
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'projection': self.projection_head(embeddings)}
    
    def setup(self, stage: str):
        """Initialize components that require num_classes from datamodule."""
        if self._num_classes is None:
            self._num_classes = self.trainer.datamodule.num_classes
        
        # Initialize proxy anchor loss if needed
        if self.loss_name == "proxy_anchor" and self.main_loss is None:
            self.main_loss = ProxyAnchorLoss(
                num_classes=self._num_classes,
                **self._proxy_anchor_params
            )
    
    def _shared_step(self, batch, batch_idx, stage: str):
        embeddings, labels = batch
        outputs = self(embeddings)
        
        total_loss = 0.0
        if self.loss_weight > 0:
            loss = self.main_loss(outputs['projection'], labels)
            total_loss += self.loss_weight * loss
            self.log(
                f'{stage}/{self.loss_name}_loss',
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )
        
        self.log(
            f'{stage}/loss',
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        return total_loss
    
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
        )
        
        scheduler_type = self.hparams.scheduler_type.lower()
        if scheduler_type == "none":
            return optimizer
        
        scheduler_params = self.hparams.scheduler_params or {}
        
        if scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_params.get("max_lr", self.hparams.learning_rate * 10),
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=scheduler_params.get("pct_start", 0.3),
                anneal_strategy=scheduler_params.get("anneal_strategy", "cos"),
                div_factor=scheduler_params.get("div_factor", 25.0),
                final_div_factor=scheduler_params.get("final_div_factor", 10000.0),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params.get("T_max", self.trainer.max_epochs),
                eta_min=scheduler_params.get("eta_min", 0.0),
            )
            
            # Add linear warmup if requested
            if self.hparams.warmup_epochs > 0:
                def warmup_lambda(epoch):
                    if epoch < self.hparams.warmup_epochs:
                        return float(epoch) / float(self.hparams.warmup_epochs)
                    return 1.0
                
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, scheduler],
                    milestones=[self.hparams.warmup_epochs]
                )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        
        return optimizer
    
    # Learning rate is logged via LearningRateMonitor callback configured in Hydra.
