import hydra
from omegaconf import DictConfig, OmegaConf

from data import RecipeDataModule
from config import Config


@hydra.main(config_name='conf/config')
def test_config(config: DictConfig):
    print(OmegaConf.to_yaml(config))

test_config()

# config = Config()
# x = RecipeDataModule(config)
# x.prepare_ingr_map()

# for y in x.test_dataloader():
    # print(y)
