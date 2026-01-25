import torch

from contrasted.losses import ProxyAnchorLoss, SupConLoss
from contrasted.model import ContrastiveModel, ProjectionHead


def test_projection_head_output_shape():
    head = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=128)
    x = torch.randn(32, 1024)
    out = head(x)
    assert out.shape == (32, 128)


def test_projection_head_normalized():
    head = ProjectionHead(input_dim=1024, hidden_dim=512, output_dim=128)
    x = torch.randn(32, 1024)
    out = head(x)
    norms = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(32), atol=1e-5)


def test_supcon_loss():
    loss_fn = SupConLoss(temperature=0.07)
    embeddings = torch.randn(16, 128)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(embeddings, labels)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_proxy_anchor_loss():
    loss_fn = ProxyAnchorLoss(num_classes=4, embedding_dim=128)
    embeddings = torch.randn(16, 128)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(embeddings, labels)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_contrastive_model_forward():
    model = ContrastiveModel(
        input_dim=1024,
        hidden_dim=512,
        output_dim=128,
        loss_type="supcon",
    )
    x = torch.randn(32, 1024)
    out = model(x)
    assert out.shape == (32, 128)
