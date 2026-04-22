"""Contrastive loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-12


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning (Khosla et al., NeurIPS 2020).

    Expects L2-normalized embeddings from ProjectionHead. Anchors whose class
    has no other member in the batch are excluded from the loss average, so the
    objective is not biased by batch composition.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        batch_size = embeddings.shape[0]
        device = embeddings.device

        similarity = (embeddings @ embeddings.T) / self.temperature
        logits = similarity - similarity.max(dim=1, keepdim=True).values.detach()

        same_class = labels.view(-1, 1).eq(labels.view(1, -1)).float()
        self_mask = torch.eye(batch_size, device=device)
        pos_mask = same_class - self_mask
        other_mask = 1.0 - self_mask

        exp_logits = torch.exp(logits) * other_mask
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(EPS)
        )

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return logits.sum() * 0.0

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
        return -mean_log_prob_pos.mean()

    def __repr__(self) -> str:
        return f"SupConLoss(temperature={self.temperature})"


class ProxyAnchorLoss(nn.Module):
    """Proxy-Anchor Loss (Kim et al., CVPR 2020).

    Expects L2-normalized embeddings from ProjectionHead. Proxies are kept
    unit-norm via an in-place renormalization at the top of ``forward``; the
    normalization is performed under ``no_grad`` so gradients flow to the raw
    proxy parameter as usual.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.1,
        alpha: float = 32.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.alpha = alpha

        proxies = torch.empty(num_classes, embedding_dim)
        nn.init.kaiming_normal_(proxies, mode="fan_out")
        proxies = F.normalize(proxies, p=2, dim=1)
        self.proxies = nn.Parameter(proxies)

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        with torch.no_grad():
            self.proxies.copy_(F.normalize(self.proxies, p=2, dim=1))

        cos_sim = embeddings @ self.proxies.T

        pos_mask = F.one_hot(labels, num_classes=self.num_classes).float()
        neg_mask = 1.0 - pos_mask

        pos_exp = torch.exp(-self.alpha * (cos_sim - self.margin))
        neg_exp = torch.exp(self.alpha * (cos_sim + self.margin))

        pos_sum = (pos_exp * pos_mask).sum(dim=0)
        neg_sum = (neg_exp * neg_mask).sum(dim=0)

        classes_with_positives = pos_mask.sum(dim=0) > 0
        num_valid = int(classes_with_positives.sum().item())

        if num_valid == 0:
            return torch.log(1.0 + neg_sum).mean()

        pos_term = torch.log(1.0 + pos_sum[classes_with_positives]).sum() / num_valid
        neg_term = torch.log(1.0 + neg_sum).sum() / self.num_classes
        return pos_term + neg_term

    def __repr__(self) -> str:
        return (
            f"ProxyAnchorLoss(num_classes={self.num_classes}, "
            f"embedding_dim={self.embedding_dim}, "
            f"margin={self.margin}, alpha={self.alpha})"
        )
