"""Contrastive loss functions for ContrasTED v1 (Center Contrastive Loss).

Full historical objective matrix lives in
``_archive/src/contrasted_losses_full.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ObjectiveOutput:
    """Structured objective result for logging component losses and diagnostics."""

    loss: Tensor
    components: dict[str, Tensor] = field(default_factory=dict)
    diagnostics: dict[str, Tensor] = field(default_factory=dict)


def _validate_proxy_hparams(
    *,
    num_classes: int,
    embedding_dim: int,
    margin: float,
    scale: float,
    label_smoothing: float,
) -> None:
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than 1.")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive.")
    if margin < 0:
        raise ValueError("margin must be non-negative.")
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be in the interval [0, 1).")


def _normalized_softmax_per_sample(
    embeddings: Tensor,
    labels: Tensor,
    bank: Tensor,
    *,
    scale: float,
    margin: float,
    label_smoothing: float,
) -> tuple[Tensor, Tensor]:
    """Per-sample CE and target cosines for a normalized proxy bank.

    ``bank`` is normalized differentiably (no in-place parameter mutation).
    Embeddings are cast to FP32 before the similarity matrix.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (batch_size, embedding_dim).")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must have shape (batch_size,).")

    num_classes = bank.shape[0]
    labels = labels.to(device=embeddings.device, dtype=torch.long)
    if labels.numel() and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError("labels must be in the range [0, num_classes).")

    centers = F.normalize(bank, p=2, dim=1)
    embeddings = embeddings.float()
    cosine = embeddings @ centers.T
    target_cosine = cosine.gather(1, labels.unsqueeze(1)).squeeze(1)

    one_hot = F.one_hot(labels, num_classes=num_classes).to(cosine.dtype)
    logits = scale * cosine - one_hot * (scale * margin)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    if label_smoothing > 0:
        off_value = label_smoothing / (num_classes - 1)
        target_probs = torch.full_like(log_probs, off_value)
        target_probs = target_probs.scatter(
            1,
            labels.unsqueeze(1),
            1.0 - label_smoothing,
        )
        contrastive = -(target_probs * log_probs).sum(dim=1)
    else:
        contrastive = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    return contrastive, target_cosine


def _init_unit_proxies(num_classes: int, embedding_dim: int) -> nn.Parameter:
    proxies = torch.empty(num_classes, embedding_dim)
    nn.init.kaiming_normal_(proxies, mode="fan_out")
    proxies = F.normalize(proxies, p=2, dim=1)
    return nn.Parameter(proxies)


class CenterContrastiveLoss(nn.Module):
    """Center Contrastive Loss (Cai et al., arXiv 2023).

    Expects L2-normalized embeddings from ProjectionHead and maintains a
    trainable, L2-normalized class center bank. This implements the paper's
    large-margin center contrast plus ``lambda_ * ||x - c_y||_2^2`` center
    constraint. For normalized vectors this is Eq. (5) plus the additive
    ``2 * lambda_`` constant that Eq. (5) drops.

    The learnable bank remains ``self.centers`` so existing checkpoints load.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.0,
        scale: float = 16.0,
        lambda_: float = 2.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        _validate_proxy_hparams(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            margin=margin,
            scale=scale,
            label_smoothing=label_smoothing,
        )
        if lambda_ < 0:
            raise ValueError("lambda_ must be non-negative.")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.lambda_ = lambda_
        self.label_smoothing = label_smoothing
        self.centers = _init_unit_proxies(num_classes, embedding_dim)

    def forward(self, embeddings: Tensor, labels: Tensor) -> ObjectiveOutput:
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings with dimension {self.embedding_dim}, "
                f"got {embeddings.shape[1]}."
            )
        contrastive, target_cosine = _normalized_softmax_per_sample(
            embeddings,
            labels,
            self.centers,
            scale=self.scale,
            margin=self.margin,
            label_smoothing=self.label_smoothing,
        )
        center_constraint = self.lambda_ * (2.0 - 2.0 * target_cosine)
        loss_h = contrastive.mean()
        loss_center = center_constraint.mean()
        loss = loss_h + loss_center
        return ObjectiveOutput(
            loss=loss,
            components={"loss_h_proxy": loss_h, "loss_center": loss_center},
        )

    def __repr__(self) -> str:
        return (
            f"CenterContrastiveLoss(num_classes={self.num_classes}, "
            f"embedding_dim={self.embedding_dim}, margin={self.margin}, "
            f"scale={self.scale}, lambda_={self.lambda_}, "
            f"label_smoothing={self.label_smoothing})"
        )
