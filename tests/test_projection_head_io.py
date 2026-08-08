from pathlib import Path

import torch

from contrasted.losses import CenterContrastiveLoss
from contrasted.model import ContrastiveModel, ProjectionHead


def test_save_load_roundtrip(tmp_path, monkeypatch):
    torch.manual_seed(0)
    head = ProjectionHead(input_dim=8, hidden_dim=6, output_dim=4, dropout=0.1)
    head.eval()

    path = head.save(tmp_path / "head.pt")
    original_load = torch.load
    weights_only_calls = []

    def tracked_load(*args, **kwargs):
        weights_only_calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    loaded = ProjectionHead.load(path)

    assert weights_only_calls == [True]
    assert (loaded.input_dim, loaded.hidden_dim, loaded.output_dim) == (8, 6, 4)
    assert loaded.dropout == 0.1

    x = torch.randn(5, 8)
    with torch.no_grad():
        assert torch.allclose(head(x), loaded(x))


def test_load_from_lightning_checkpoint(tmp_path, monkeypatch):
    torch.manual_seed(1)
    model = ContrastiveModel(
        projection_head=ProjectionHead(input_dim=8, hidden_dim=6, output_dim=4),
        loss=CenterContrastiveLoss(
            num_classes=3,
            embedding_dim=4,
            margin=0.0,
            scale=16.0,
            lambda_=0.0,
            label_smoothing=0.0,
        ),
    )
    model.eval()

    ckpt = tmp_path / "lightning.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": {"dropout": 0.0},
            "legacy_path": Path("checkpoints/run"),
        },
        ckpt,
    )

    original_load = torch.load
    weights_only_calls = []

    def tracked_load(*args, **kwargs):
        weights_only_calls.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    loaded = ProjectionHead.load(ckpt)

    assert weights_only_calls == [True, False]
    x = torch.randn(5, 8)
    with torch.no_grad():
        assert torch.allclose(model.projection_head(x), loaded(x))
