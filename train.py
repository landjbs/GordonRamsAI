import os
import hydra
from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from data import RecipeDataModule
from model import Model
from trainer import Trainer


# @hydra.main(config_name='conf/config')
# def test_config(config: DictConfig):
#     os.chdir(hydra.utils.get_original_cwd())
#     #
#     data = RecipeDataModule(config)
#     data.setup()
#     #
#     model = Model(config, data.vocab_size)
#     for x in data.train_dataloader():
#         print(x)



@hydra.main(config_name='conf/config')
def train(config: DictConfig, **kwargs) -> None:
    # reset to og cwd to allow relative paths
    os.chdir(hydra.utils.get_original_cwd())
    # set seeds
    seed_everything(0)
    # build data module
    # data_module = RecipeDataModule(config)
    # data_module.setup()
    # build model
    model = Model(config, data_module.vocab_size)
    # build trainer
    trainer = Trainer.from_config(config)
    # run training loop
    trainer.fit(model)
        # train_dataloader=data_module.train_dataloader(),
        # val_dataloaders=data_module.val_dataloader()
    # )



if __name__ == '__main__':
    train()
