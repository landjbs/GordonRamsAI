import os
import hydra
from omegaconf import DictConfig, OmegaConf

from data import RecipeDataModule
from model import Model


@hydra.main(config_name='conf/config')
def test_config(config: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    model = Model(config, 1) #data.vocab_size)
    print(model)
    data = RecipeDataModule(config)
    data.setup()


if __name__ == '__main__':
    test_config()
