"""Controlled multi-stream initialization for reproducible training runs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


def component_seeds(seed: int) -> dict[str, int]:
    """Derive fixed offset seeds for each random stream."""
    return {
        "head": seed + 1000,
        "proxy_h": seed + 2000,
        "sampler": seed + 3000,
        "training": seed + 4000,
    }


@contextmanager
def seeded_rng(seed: int) -> Iterator[None]:
    """Temporarily fork RNGs and seed them for one component."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Stable SHA256 of a CPU float32 contiguous tensor."""
    arr = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def init_unit_parameter(
    parameter: nn.Parameter,
    *,
    seed: int,
) -> str:
    """Kaiming-normal + L2-normalize ``parameter`` under ``seed``; return hash."""
    with seeded_rng(seed):
        nn.init.kaiming_normal_(parameter, mode="fan_out")
        with torch.no_grad():
            parameter.copy_(F.normalize(parameter, p=2, dim=1))
    return tensor_sha256(parameter.data)


def init_projection_head(head: nn.Module, *, seed: int) -> str:
    """Re-initialize modules in ``head`` under ``seed``; return hash."""
    with seeded_rng(seed):
        for module in head.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
    chunks: list[bytes] = []
    for name, param in head.named_parameters():
        chunks.append(name.encode())
        chunks.append(param.detach().float().cpu().contiguous().numpy().tobytes())
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def apply_controlled_initialization(
    model: nn.Module,
    *,
    seed: int,
) -> dict[str, str]:
    """Initialize projection head and CCL centers on independent streams."""
    seeds = component_seeds(seed)
    hashes: dict[str, str] = {}

    head = model.get_submodule("projection_head")
    hashes["projection_head"] = init_projection_head(head, seed=seeds["head"])

    loss = model.get_submodule("loss")
    centers = getattr(loss, "centers", None)
    if isinstance(centers, nn.Parameter):
        hashes["proxy_h"] = init_unit_parameter(centers, seed=seeds["proxy_h"])

    hashes["seed"] = str(seed)
    hashes["head_seed"] = str(seeds["head"])
    hashes["proxy_h_seed"] = str(seeds["proxy_h"])
    hashes["sampler_seed"] = str(seeds["sampler"])
    hashes["training_seed"] = str(seeds["training"])
    return hashes
