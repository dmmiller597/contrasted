import numpy as np
import pytest
import torch

from contrasted.losses import CenterContrastiveLoss
from contrasted.model import ProjectionHead, project


def test_projection_head_defaults_to_concat_input_dim():
    head = ProjectionHead()
    assert head.input_dim == 2048


def test_projection_head_output_shape_and_normalized():
    head = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=128)
    out = head(torch.randn(32, 1024))
    assert out.shape == (32, 128)
    norms = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(32), atol=1e-5)


def test_projection_head_accepts_concat_input_dim():
    head = ProjectionHead(input_dim=2048, hidden_dim=512, output_dim=128)
    out = head(torch.randn(8, 2048))
    assert out.shape == (8, 128)
    norms = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_project_rejects_store_dim_mismatch():
    head = ProjectionHead(input_dim=2048, hidden_dim=32, output_dim=8, dropout=0.0)
    aa_only = np.zeros((4, 1024), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match projection head"):
        project(head, aa_only, device="cpu")


def test_center_contrastive_loss():
    loss_fn = CenterContrastiveLoss(num_classes=4, embedding_dim=128)
    embeddings = torch.nn.functional.normalize(torch.randn(16, 128), dim=1)
    embeddings.requires_grad_()
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3])

    out = loss_fn(embeddings, labels)
    out.loss.backward()

    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert embeddings.grad is not None
    assert loss_fn.centers.grad is not None
    assert "centers" in loss_fn.state_dict()


def test_center_contrastive_loss_matches_paper_objective():
    loss_fn = CenterContrastiveLoss(
        num_classes=3,
        embedding_dim=2,
        margin=0.2,
        scale=16.0,
        lambda_=1.5,
        label_smoothing=0.0,
    )
    embeddings = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.5], [-0.2, 1.0], [0.3, -0.7]]),
        dim=1,
    )
    centers = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]),
        dim=1,
    )
    labels = torch.tensor([0, 1, 2])

    with torch.no_grad():
        loss_fn.centers.copy_(centers)

    cosine = embeddings @ centers.T
    batch_indices = torch.arange(embeddings.shape[0])
    target_cosine = cosine[batch_indices, labels]
    logits = loss_fn.scale * cosine
    logits[batch_indices, labels] -= loss_fn.scale * loss_fn.margin
    contrastive = torch.logsumexp(logits, dim=1) - loss_fn.scale * (
        target_cosine - loss_fn.margin
    )
    center_constraint = loss_fn.lambda_ * (2.0 - 2.0 * target_cosine)
    expected = (contrastive + center_constraint).mean()

    assert torch.allclose(loss_fn(embeddings, labels).loss, expected)
