import hydra
from omegaconf import DictConfig
import lightning as L
from hydra.utils import instantiate
from contrasted.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    set_seed(cfg.seed, deterministic=cfg.trainer.deterministic)
    
    # Instantiate datamodule and model via Hydra
    datamodule = instantiate(cfg.datamodule)
    model = instantiate(cfg.model)
    
    # Instantiate logger, callbacks, and trainer
    logger = instantiate(cfg.logging.logger)
    callbacks = [instantiate(cb) for cb in cfg.logging.callbacks]
    trainer = L.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)
    
    trainer.fit(model, datamodule=datamodule)
    if cfg.run.do_test_after_fit:
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.run.test_ckpt_path)


if __name__ == "__main__":
    main()
