"""Train contrastive model on CATH embeddings."""

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from contrasted.utils import set_seed


def run(cfg: DictConfig) -> None:
    set_seed(cfg.seed, deterministic=cfg.trainer.deterministic)

    datamodule = instantiate(cfg.datamodule)

    # Resolve num_classes for losses that need it (e.g. ProxyAnchorLoss).
    loss_cfg = OmegaConf.select(cfg, "model.loss")
    if loss_cfg is not None and "num_classes" in loss_cfg:
        datamodule.setup("fit")
        with open_dict(cfg):
            cfg.model.loss.num_classes = datamodule.num_classes

    model = instantiate(cfg.model)

    logger = instantiate(cfg.logger)
    callbacks = [instantiate(cb) for cb in cfg.callbacks]
    trainer = L.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI wrapper
    run(cfg)


if __name__ == "__main__":
    main()
