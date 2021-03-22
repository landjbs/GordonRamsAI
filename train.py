import os
import hydra
from omegaconf import DictConfig, OmegaConf

from data import RecipeDataModule
from model import Model


@hydra.main(config_name='conf/config')
def test_config(config: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    # 
    data = RecipeDataModule(config)
    data.setup()
    #
    model = Model(config, data.vocab_size)
    for x in data.train_dataloader():
        print(model(x))


if __name__ == '__main__':
    test_config()
