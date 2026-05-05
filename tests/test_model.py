import torch

from contrasted.losses import ProxyAnchorLoss, SupConLoss
from contrasted.model import ProjectionHead


def test_projection_head_output_shape_and_normalized():
    head = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=128)
    out = head(torch.randn(32, 1024))
    assert out.shape == (32, 128)
    norms = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(32), atol=1e-5)


def test_supcon_loss():
    loss_fn = SupConLoss(temperature=0.07)
    embeddings = torch.nn.functional.normalize(torch.randn(16, 128), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(embeddings, labels)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_supcon_loss_all_singletons_zero():
    """A batch with no positive pairs should yield zero loss (not NaN)."""
    loss_fn = SupConLoss(temperature=0.07)
    embeddings = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)
    labels = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(embeddings, labels)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_supcon_loss_finite_with_mixed_singletons():
    """Mixed batch (some classes singletons, some not) gives a finite loss."""
    loss_fn = SupConLoss(temperature=0.07)
    embeddings = torch.nn.functional.normalize(torch.randn(6, 128), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 3])
    loss = loss_fn(embeddings, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_proxy_anchor_loss():
    loss_fn = ProxyAnchorLoss(num_classes=4, embedding_dim=128)
    embeddings = torch.nn.functional.normalize(torch.randn(16, 128), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(embeddings, labels)
    assert loss.ndim == 0
    assert loss.item() > 0
