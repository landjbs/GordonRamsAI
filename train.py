import os
import hydra
from omegaconf import DictConfig, OmegaConf

from data import RecipeDataModule
from model import Model


@hydra.main(config_name='conf/config')
def test_config(config: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    x = RecipeDataModule(config)
    x.setup()
    model = Model(config)
    print(model)


if __name__ == '__main__':
    test_config()
