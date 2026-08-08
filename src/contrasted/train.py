"""Train contrastive model on CATH embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from contrasted.init_control import apply_controlled_initialization, component_seeds
from contrasted.utils import set_seed


def _inject_loss_class_counts(cfg: DictConfig, datamodule) -> None:
    """Fill the runtime class count on the loss config from the datamodule."""
    loss_cfg = OmegaConf.select(cfg, "model.loss")
    if loss_cfg is None:
        return
    with open_dict(cfg):
        if "num_classes" in loss_cfg:
            cfg.model.loss.num_classes = datamodule.num_classes


def _write_run_provenance(cfg: DictConfig, init_hashes: dict[str, str]) -> None:
    run_dir = Path(cfg.trainer.get("default_root_dir") or ".")
    # Prefer Hydra output dir when available.
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            run_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        pass
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": int(cfg.seed),
        "objective_id": cfg.get("objective_id"),
        "sweep_parameter": cfg.get("sweep_parameter"),
        "sweep_value": cfg.get("sweep_value"),
        "run_stage": cfg.get("run_stage"),
        "init_hashes": init_hashes,
        "component_seeds": component_seeds(int(cfg.seed)),
        "test_after_fit": bool(cfg.get("test_after_fit", True)),
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (run_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg),
        encoding="utf-8",
        newline="\n",
    )


def run(cfg: DictConfig) -> None:
    """Construct the datamodule, model, and trainer."""
    seed = int(cfg.seed)
    set_seed(seed, deterministic=cfg.trainer.deterministic)

    # Set up dense training labels and optional split checks.
    datamodule = instantiate(cfg.datamodule)
    if cfg.datamodule.get("sampler_seed") is None and cfg.datamodule.get(
        "balanced_sampler", False
    ):
        with open_dict(cfg):
            cfg.datamodule.sampler_seed = component_seeds(seed)["sampler"]
        datamodule.sampler_seed = cfg.datamodule.sampler_seed
    datamodule.setup("fit")

    # Resolve the dense H class count into the loss config.
    _inject_loss_class_counts(cfg, datamodule)

    model = instantiate(cfg.model)

    # Controlled multi-stream initialization and provenance.
    init_hashes: dict[str, str] = {}
    if cfg.get("controlled_init", False):
        init_hashes = apply_controlled_initialization(model, seed=seed)
    _write_run_provenance(cfg, init_hashes)

    logger = instantiate(cfg.logger) if cfg.get("logger") else False
    callbacks = [instantiate(cb) for cb in cfg.callbacks]
    trainer = L.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    # Reset global RNG to the training stream before fit.
    training_seed = component_seeds(seed)["training"]
    if cfg.get("controlled_init", False):
        set_seed(training_seed, deterministic=cfg.trainer.deterministic)
    else:
        set_seed(seed, deterministic=cfg.trainer.deterministic)

    trainer.fit(model, datamodule=datamodule)
    if cfg.get("test_after_fit", True):
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


@hydra.main(version_base=None, config_path="pkg://configs", config_name="train")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI wrapper
    run(cfg)


if __name__ == "__main__":
    main()
