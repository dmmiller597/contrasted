"""Contrastive loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-12


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning (Khosla et al., NeurIPS 2020).

    Expects L2-normalized embeddings from ProjectionHead.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        device = embeddings.device
        batch_size = embeddings.shape[0]

        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp(min=EPS)
        )

        pos_mask_sum = mask.sum(dim=1).clamp(min=EPS)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_mask_sum

        return -mean_log_prob_pos.mean()

    def __repr__(self) -> str:
        return f"SupConLoss(temperature={self.temperature})"


class ProxyAnchorLoss(nn.Module):
    """Proxy-Anchor Loss (Kim et al., CVPR 2020).

    Expects L2-normalized embeddings from ProjectionHead.
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

        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        device = embeddings.device

        proxies_norm = F.normalize(self.proxies, p=2, dim=1)
        cos_sim = F.linear(embeddings, proxies_norm)

        P_one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(device)
        N_one_hot = 1.0 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos_sim - self.margin))
        neg_exp = torch.exp(self.alpha * (cos_sim + self.margin))

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(
            dim=0
        )
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(
            dim=0
        )

        with_pos_proxies = torch.nonzero(
            P_one_hot.sum(dim=0) > 0, as_tuple=False
        ).squeeze(dim=1)
        num_valid_pos_proxies = with_pos_proxies.numel()

        if num_valid_pos_proxies == 0:
            return torch.log(1.0 + N_sim_sum).mean()

        pos_term = (
            torch.log(1.0 + P_sim_sum[with_pos_proxies]).sum() / num_valid_pos_proxies
        )
        neg_term = torch.log(1.0 + N_sim_sum).sum() / self.num_classes

        return pos_term + neg_term

    def __repr__(self) -> str:
        return (
            f"ProxyAnchorLoss(num_classes={self.num_classes}, "
            f"embedding_dim={self.embedding_dim}, "
            f"margin={self.margin}, alpha={self.alpha})"
        )
