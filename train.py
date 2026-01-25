"""Train contrastive model on CATH embeddings."""

import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig

from contrasted.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    set_seed(cfg.seed, deterministic=cfg.trainer.deterministic)

    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)

    logger = instantiate(cfg.logger)
    callbacks = [instantiate(cb) for cb in cfg.callbacks]
    trainer = L.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best", weights_only=False)


if __name__ == "__main__":
    main()
