import os
import hydra
from omegaconf import DictConfig, OmegaConf

from data import RecipeDataModule


@hydra.main(config_name='conf/config')
def test_config(config: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    x = RecipeDataModule(config)
    x.setup()


if __name__ == '__main__':
    test_config()
