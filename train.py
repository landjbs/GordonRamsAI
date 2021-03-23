import os
import hydra
from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from data import RecipeDataModule
from model import Model
from trainer import Trainer


@hydra.main(config_name='conf/config')
def train(config: DictConfig, **kwargs) -> None:
    # reset to og cwd to allow relative paths
    os.chdir(hydra.utils.get_original_cwd())
    # set seeds
    seed_everything(0)
    # build model
    model = Model(config)
    # build trainer
    trainer = Trainer.from_config(config)
    # run training loop
    trainer.fit(model)



if __name__ == '__main__':
    train()
