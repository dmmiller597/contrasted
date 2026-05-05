"""Checkpoint loading helpers for inference entrypoints."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from contrasted.losses import ProxyAnchorLoss, SupConLoss
from contrasted.model import ContrastiveModel, ProjectionHead


class _InferenceOnlyLoss(nn.Module):
    """Placeholder loss for checkpoints that only need projection weights."""

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "This checkpoint was loaded for inference and does not include enough "
            "metadata to reconstruct a training loss."
        )


def load_model_for_inference(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> ContrastiveModel:
    """Load a :class:`ContrastiveModel` for projection-only inference.

    Training injects the projection head and loss modules, so Lightning cannot
    reconstruct the module from checkpoint hyperparameters alone. For inference
    CLIs we only need the projection head; infer its dimensions from the saved
    tensor shapes and reconstruct the loss only when the checkpoint contains
    enough metadata/state to do so.
    """

    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint has no state_dict: {checkpoint_path}")

    hparams = checkpoint.get("hyper_parameters") or {}
    if not isinstance(hparams, dict):
        hparams = {}

    projection_head = _build_projection_head(state_dict, hparams)
    output_dim = int(state_dict["projection_head.net.4.weight"].shape[0])
    loss = _build_loss(state_dict, hparams, output_dim)
    model = ContrastiveModel(
        projection_head=projection_head,
        loss=loss,
        learning_rate=float(hparams.get("learning_rate", 1e-3)),
        weight_decay=float(hparams.get("weight_decay", 1e-4)),
        max_epochs=int(hparams.get("max_epochs", 200)),
        warmup_epochs=int(hparams.get("warmup_epochs", 10)),
        min_lr=float(hparams.get("min_lr", 1e-6)),
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _build_projection_head(
    state_dict: dict[str, torch.Tensor], hparams: dict[str, Any]
) -> ProjectionHead:
    first_weight = state_dict.get("projection_head.net.0.weight")
    final_weight = state_dict.get("projection_head.net.4.weight")
    if first_weight is None or final_weight is None:
        raise ValueError(
            "Checkpoint is missing projection_head.net.0.weight or "
            "projection_head.net.4.weight."
        )

    hidden_dim, input_dim = first_weight.shape
    output_dim, final_hidden_dim = final_weight.shape
    if hidden_dim != final_hidden_dim:
        raise ValueError(
            "Checkpoint projection head has inconsistent hidden dimensions: "
            f"{hidden_dim} != {final_hidden_dim}."
        )

    return ProjectionHead(
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        output_dim=int(output_dim),
        dropout=float(hparams.get("dropout", 0.0)),
    )


def _build_loss(
    state_dict: dict[str, torch.Tensor],
    hparams: dict[str, Any],
    output_dim: int,
) -> nn.Module:
    loss_params = hparams.get("loss_params") or {}
    if not isinstance(loss_params, dict):
        loss_params = {}

    proxies = state_dict.get("loss.proxies")
    if proxies is not None:
        num_classes, embedding_dim = proxies.shape
        return ProxyAnchorLoss(
            num_classes=int(num_classes),
            embedding_dim=int(embedding_dim),
            margin=float(loss_params.get("margin", hparams.get("margin", 0.1))),
            alpha=float(loss_params.get("alpha", hparams.get("alpha", 32.0))),
        )

    loss_type = str(hparams.get("loss_type", "")).lower()
    if loss_type == "supcon":
        return SupConLoss(
            temperature=float(
                loss_params.get("temperature", hparams.get("temperature", 0.07))
            )
        )

    num_classes = hparams.get("num_classes")
    if loss_type == "proxy_anchor" and num_classes is not None:
        return ProxyAnchorLoss(
            num_classes=int(num_classes),
            embedding_dim=output_dim,
            margin=float(loss_params.get("margin", hparams.get("margin", 0.1))),
            alpha=float(loss_params.get("alpha", hparams.get("alpha", 32.0))),
        )

    return _InferenceOnlyLoss()
